clear; clc;

% Keep FAST_MODE enabled for the first end-to-end run. After the script
% finishes successfully, set it to false for the expensive formal study.
FAST_MODE = true;

DATASET_NAME = 'Handwritten';
PER_LIST = [0.0, 0.1, 0.3, 0.5];
GRID_TARGET_PER = 0.3;
PN_NEIGHBORS = 15;
KMEANS_REPS = 20;

if FAST_MODE
    NUM_RUNS = 1;
    T_MAX = 3;
    ANCHOR_M = 30;
    ANCHOR_S = 3;
    R_ANCHOR = 4;
    LG_SPACE = 0.5;
    LA_SPACE = 0.1;
    Z_MAX_ITER = 3;
    Z_MAX_SECONDS = 5;
    run_profile = 'FAST VALIDATION';
else
    NUM_RUNS = 10;
    T_MAX = 200;
    ANCHOR_M = 300;
    ANCHOR_S = 3;
    R_ANCHOR = 10;
    LG_SPACE = [0.1, 0.3, 0.5, 1.0, 1.5];
    LA_SPACE = [0.01, 0.05, 0.1, 0.3, 0.5];
    Z_MAX_ITER = 20;
    Z_MAX_SECONDS = 10;
    run_profile = 'FORMAL FULL STUDY';
end

fprintf('======================================================\n');
fprintf('  SR-FIMVC HANDWRITTEN EXPERIMENT\n');
fprintf('  PROFILE: %s\n', run_profile);
fprintf('  RUNS=%d, T=%d, ANCHORS=%d, KMEANS_REPS=%d\n', ...
        NUM_RUNS, T_MAX, ANCHOR_M, KMEANS_REPS);
fprintf('  GRID: %d x %d parameter combinations\n', ...
        numel(LG_SPACE), numel(LA_SPACE));
fprintf('======================================================\n');
drawnow;

[Xlist, gt, k, ~] = load_dataset_by_name(DATASET_NAME);

config_base = struct( ...
    'T', T_MAX, ...
    'use_anchor', true, ...
    'anchor_m', ANCHOR_M, ...
    'anchor_s', ANCHOR_S, ...
    'pn', PN_NEIGHBORS, ...
    'R_anchor', R_ANCHOR, ...
    'R_comm', 3, ...
    'warm_start_iter', 50, ...
    'smart_axis', true, ...
    'metric', 'cosine', ...
    'kmeans_reps', KMEANS_REPS, ...
    'structure_z_enabled', true, ...
    'lambda1', 1.0, ...
    'lambda2', 1.0, ...
    'z_max_iter', Z_MAX_ITER, ...
    'z_max_seconds', Z_MAX_SECONDS, ...
    'z_constraint_tol', 1e-7, ...
    'h_gradient_mode', 'objective', ...
    'h_monotone_safeguard', true, ...
    'completion_init_mode', 'mean', ...
    'local_approx_mode', 'orthogonal_bcd', ...
    'local_approx_max_iter', 50, ...
    'server_consensus_mode', 'orthogonal_bcd', ...
    'server_consensus_max_iter', 50);

config_base.X_raw = Xlist;
config_base.gt_raw = gt;
config_base.k = k;

fprintf('\n======================================================\n');
fprintf('  PHASE 1: PARAMETER SEARCH AT PER=%.1f\n', GRID_TARGET_PER);
fprintf('======================================================\n');

best_acc = -inf;
best_lg = NaN;
best_la = NaN;
grid_total = numel(LG_SPACE) * numel(LA_SPACE);
grid_index = 0;

for lg = LG_SPACE
    for la = LA_SPACE
        grid_index = grid_index + 1;
        fprintf('\n[Grid %d/%d] lambda_graph=%.3g, lambda_align=%.3g\n', ...
                grid_index, grid_total, lg, la);
        fprintf('The first round can take time because six constrained Z updates run before Round 1 is printed.\n');
        drawnow;

        tmp_cfg = config_base;
        tmp_cfg.PER = GRID_TARGET_PER;
        tmp_cfg.lambda_graph = lg;
        tmp_cfg.lambda_align = la;

        trial_timer = tic;
        try
            R = main_fed_SR_FIMVC(DATASET_NAME, 1, tmp_cfg);
            fprintf('[Grid %d/%d completed in %.1f s] ACC=%.4f, NMI=%.4f, ARI=%.4f\n', ...
                    grid_index, grid_total, toc(trial_timer), ...
                    R.best.acc, R.best.nmi, R.best.ari);
            if R.best.acc > best_acc
                best_acc = R.best.acc;
                best_lg = lg;
                best_la = la;
            end
        catch ME
            warning('Grid trial %d failed: %s', grid_index, ME.message);
        end
        drawnow;
    end
end

if ~isfinite(best_acc)
    error('All grid-search trials failed. Review the warnings above.');
end

fprintf('\nBest grid result: ACC=%.4f, lambda_graph=%.3g, lambda_align=%.3g\n', ...
        best_acc, best_lg, best_la);

config_base.lambda_graph = best_lg;
config_base.lambda_align = best_la;

fprintf('\n======================================================\n');
fprintf('  PHASE 2: FINAL EVALUATION\n');
fprintf('======================================================\n');

results_table = table();
all_runs_details = struct('PER', {}, 'RUN_ID', {}, 'R', {});
detail_counter = 1;
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
checkpoint_filename = sprintf('checkpoint_%s_%s.mat', DATASET_NAME, timestamp);
save_filename = sprintf('results_FINAL_%s_%s.mat', DATASET_NAME, timestamp);

for p = 1:numel(PER_LIST)
    current_per = PER_LIST(p);
    acc_runs = zeros(NUM_RUNS, 1);
    nmi_runs = zeros(NUM_RUNS, 1);
    ari_runs = zeros(NUM_RUNS, 1);
    purity_runs = zeros(NUM_RUNS, 1);

    fprintf('\n--- PER=%.2f: %d run(s) ---\n', current_per, NUM_RUNS);
    drawnow;

    for run = 1:NUM_RUNS
        fprintf('[PER %.2f, run %d/%d] starting...\n', current_per, run, NUM_RUNS);
        drawnow;

        config = config_base;
        config.PER = current_per;
        run_timer = tic;
        R = main_fed_SR_FIMVC(DATASET_NAME, run, config);

        acc_runs(run) = R.best.acc;
        nmi_runs(run) = R.best.nmi;
        ari_runs(run) = R.best.ari;
        purity_runs(run) = R.best.purity;

        all_runs_details(detail_counter).PER = current_per;
        all_runs_details(detail_counter).RUN_ID = run;
        all_runs_details(detail_counter).R = R;
        detail_counter = detail_counter + 1;

        fprintf('[PER %.2f, run %d/%d completed in %.1f s] ACC=%.4f, NMI=%.4f, ARI=%.4f\n', ...
                current_per, run, NUM_RUNS, toc(run_timer), ...
                R.best.acc, R.best.nmi, R.best.ari);
        save(checkpoint_filename, 'results_table', 'all_runs_details', ...
             'config_base', 'FAST_MODE', '-v7.3');
        drawnow;
    end

    new_row = table(current_per, mean(acc_runs), std(acc_runs), ...
                    mean(nmi_runs), std(nmi_runs), ...
                    mean(ari_runs), std(ari_runs), ...
                    mean(purity_runs), std(purity_runs), ...
        'VariableNames', {'PER', 'ACC_Mean', 'ACC_Std', ...
                          'NMI_Mean', 'NMI_Std', ...
                          'ARI_Mean', 'ARI_Std', ...
                          'Pur_Mean', 'Pur_Std'});
    results_table = [results_table; new_row]; %#ok<AGROW>
    save(checkpoint_filename, 'results_table', 'all_runs_details', ...
         'config_base', 'FAST_MODE', '-v7.3');
end

fprintf('\n======================================================\n');
fprintf('  FINAL RESULTS\n');
fprintf('======================================================\n');
disp(results_table);

save(save_filename, 'results_table', 'all_runs_details', ...
     'config_base', 'FAST_MODE', '-v7.3');
fprintf('Saved final results to: %s\n', save_filename);
fprintf('Checkpoint file: %s\n', checkpoint_filename);

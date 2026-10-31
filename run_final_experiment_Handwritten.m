clear; clc;
script_root=fileparts(mfilename('fullpath'));
addpath(script_root,'-begin');

% Keep FAST_MODE enabled for the first end-to-end run. After the script
% finishes successfully, set it to false for the expensive formal study.
FAST_MODE = true;

DATASET_NAME = 'Handwritten';
PER_LIST = [0.0, 0.1, 0.3, 0.5];
PN_NEIGHBORS = 15;
KMEANS_REPS = 20;
LAMBDA_GRAPH = 0.5;
LAMBDA_ALIGN = 0.1;

if FAST_MODE
    NUM_RUNS = 1;
    T_MAX = 3;
    ANCHOR_M = 30;
    ANCHOR_S = 3;
    R_ANCHOR = 4;
    Z_PROJECTED_MAX_ITER = 3;
    run_profile = 'FAST VALIDATION';
else
    NUM_RUNS = 10;
    T_MAX = 200;
    ANCHOR_M = 300;
    ANCHOR_S = 3;
    R_ANCHOR = 10;
    Z_PROJECTED_MAX_ITER = 5;
    run_profile = 'FORMAL FULL STUDY';
end

fprintf('======================================================\n');
fprintf('  SR-FIMVC HANDWRITTEN EXPERIMENT\n');
fprintf('  PROFILE: %s\n', run_profile);
fprintf('  RUNS=%d, T=%d, ANCHORS=%d, KMEANS_REPS=%d\n', ...
        NUM_RUNS, T_MAX, ANCHOR_M, KMEANS_REPS);
fprintf('  FIXED PARAMETERS: lambda_graph=%.3g, lambda_align=%.3g\n', ...
        LAMBDA_GRAPH, LAMBDA_ALIGN);
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
    'smart_axis', false, ...
    'metric', 'cosine', ...
    'kmeans_reps', KMEANS_REPS, ...
    'structure_z_enabled', true, ...
    'z_solver', 'projected', ...
    'lambda1', 1.0, ...
    'lambda2', 1.0, ...
    'z_projected_max_iter', Z_PROJECTED_MAX_ITER, ...
    'z_projected_max_backtracks', 12, ...
    'z_projected_tol', 1e-6, ...
    'h_gradient_mode', 'objective', ...
    'h_monotone_safeguard', true, ...
    'completion_init_mode', 'mean', ...
    'local_approx_mode', 'orthogonal_bcd', ...
    'local_approx_max_iter', 50, ...
    'server_consensus_mode', 'orthogonal_bcd', ...
    'server_consensus_max_iter', 50, ...
    'lambda_graph', LAMBDA_GRAPH, ...
    'lambda_align', LAMBDA_ALIGN);

config_base.X_raw = Xlist;
config_base.gt_raw = gt;
config_base.k = k;

fprintf('\n======================================================\n');
fprintf('  FINAL EVALUATION (NO GROUND-TRUTH MODEL SELECTION)\n');
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

        acc_runs(run) = R.final.acc;
        nmi_runs(run) = R.final.nmi;
        ari_runs(run) = R.final.ari;
        purity_runs(run) = R.final.purity;

        all_runs_details(detail_counter).PER = current_per;
        all_runs_details(detail_counter).RUN_ID = run;
        all_runs_details(detail_counter).R = R;
        detail_counter = detail_counter + 1;

        fprintf('[PER %.2f, run %d/%d completed in %.1f s] ACC=%.4f, NMI=%.4f, ARI=%.4f\n', ...
                current_per, run, NUM_RUNS, toc(run_timer), ...
                R.final.acc, R.final.nmi, R.final.ari);
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

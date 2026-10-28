function R = main_fed_SR_FIMVC(dataset_name, run, config)
% Structural-Z reference revision of the corrected baseline.
% See README.md for constraints, solver dependencies and validation limits.

rng(run);

%% ---------- defaults ----------
df = struct( ...
    'k',[], 'T',10, ...
    'lambda_graph',0.5, 'lambda_align',0.1, 'lambda2',1.0, ... 
    'structure_z_enabled',true, 'lambda1',1.0, ...
    'z_max_iter',20, 'z_max_seconds',10, 'z_constraint_tol',1e-7, ...
    'h_gradient_mode','objective', 'h_monotone_safeguard',true, ...
    'completion_init_mode','mean', ...
    'local_approx_mode','orthogonal_bcd','local_approx_max_iter',50, ...
    'server_consensus_mode','orthogonal_bcd','server_consensus_max_iter',50, ...
    'use_anchor',true, 'pn',10, ...
    'R_comm',1, 'R_anchor',1, ...
    'anchor_m',[], 'anchor_s',12, ...
    'smart_axis',false, 'max_feat',[], ...
    'auto_k',true, 'force_k',false, ...
    'gt_strict', true, 'data_dir','data', ...
    'PER', 0.3, 'metric', 'euclidean', ...
    'mu_ridge',1e-3, 'eps_w',1e-6, ...       
    'warm_start_iter', 50, ...
    'kmeans_reps', 5);

config = set_defaults(config, df);

%% ---------- load data ----------
t0 = tic;
folds = struct('PER', config.PER);

if strcmpi(dataset_name, 'orl')
    matPath = fullfile(config.data_dir, [dataset_name, '.mat']);
    if ~exist(matPath, 'file'), matPath = fullfile(config.data_dir, [lower(dataset_name), '.mat']); end
    if ~exist(matPath, 'file'), matPath = fullfile(config.data_dir, [upper(dataset_name), '.mat']); end
    S = load(matPath);
    if isfield(S,'Xlist'), X_raw=S.Xlist; gt_raw=S.gt_final; 
    elseif isfield(S,'X'), X_raw=S.X; gt_raw=S.gt; 
    else, X_raw=S.data; gt_raw=S.label; end
    [X_full_list, gt, k, Mlist] = load_data_bypass(X_raw, gt_raw, folds, config.data_dir);
else
    [X_full_list, gt, k, Mlist] = load_dataset_by_name(dataset_name, folds, config.data_dir);
end

if isfield(config, 'X_raw') && ~isempty(config.X_raw)
    X_full_list = config.X_raw; gt = config.gt_raw;
    if isvector(gt), k = numel(unique(gt)); end
end

gt_vec = normalize_gt(gt, config.gt_strict);
if ~isempty(gt) && ~isinteger(gt), gt = grp2idx(gt); end
gt_vec = grp2idx(gt_vec);
rng(run);

%% ---------- Initialization ----------

if isfield(config,'smart_axis') && config.smart_axis
    for v_std = 1:numel(X_full_list)
        Xv = X_full_list{v_std};
        rownorm = sqrt(sum(Xv.^2, 2)) + 1e-12;
        X_full_list{v_std} = Xv ./ rownorm;
    end
else
    for v_std = 1:numel(X_full_list)
        X_full_list{v_std} = zscore(X_full_list{v_std}, 0, 1);
    end
end

% =========================================================================

% =========================================================================
if isfield(config, 'sigma') && config.sigma > 0
    for v_noise = 1:numel(X_full_list)

        X_full_list{v_noise} = X_full_list{v_noise} + config.sigma * randn(size(X_full_list{v_noise}));
    end
end

if isfield(config, 'rho') && config.rho > 0
    for v_outlier = 1:numel(X_full_list)
        [num_samples, num_feats] = size(X_full_list{v_outlier});

        num_outliers = round(num_samples * config.rho);
        if num_outliers > 0

            outlier_idx = randperm(num_samples, num_outliers);
            X_full_list{v_outlier}(outlier_idx, :) = 3.0 * randn(num_outliers, num_feats);
        end
    end
end
% =========================================================================


[X_incomplete_list, Omega_list] = create_incomplete_data(X_full_list, config.PER);

X_completed_list = cell(size(X_incomplete_list));
for v = 1:numel(X_incomplete_list)
    switch lower(char(config.completion_init_mode))
        case 'mean'
            X_completed_list{v} = mean_fill_missing(X_incomplete_list{v},Omega_list{v});
        case 'zero'
            X_completed_list{v} = X_incomplete_list{v} .* Omega_list{v};
        otherwise
            error('main_fed_SR_FIMVC:InvalidCompletionInit', ...
                  'completion_init_mode must be mean or zero.');
    end
end

N  = size(X_completed_list{1},1);
nV = numel(X_completed_list);
config.k = max(2, numel(unique(gt_vec)));

%% ---------- SVD + Global Init (Hybrid Version) ----------




if ~isfield(config, 'use_raw_init'), config.use_raw_init = false; end
if ~isfield(config, 'kmeans_reps'), config.kmeans_reps = 20; end

if config.use_raw_init

    fprintf('[Init] Strategy: RAW Features Concat (Mean Imputation for Missing)...\n');
    X_concat_init = [];
    for v = 1:nV
        X_concat_init = [X_concat_init, X_completed_list{v}];
    end
    

    Hv = cell(1,nV); H_init_info = cell(1,nV);
    for v=1:nV
        [Hv{v}, ~, H_init_info{v}] = ...
            initialize_latent_view(X_completed_list{v}, config.k);
    end
else

    fprintf('[Init] Strategy: SVD Dimension Reduction...\n');
    Hv = cell(1,nV); H_cluster_init = cell(1,nV);
    H_init_info = cell(1,nV);
    for v = 1:nV
        [Hv{v}, H_cluster_init{v}, H_init_info{v}] = ...
            initialize_latent_view(X_completed_list{v}, config.k);
    end


    X_concat_init = [];
    for v = 1:nV, X_concat_init = [X_concat_init, H_cluster_init{v}]; end
end


fprintf('       > K-Means: Reps=%d, DataDim=%d\n', config.kmeans_reps, size(X_concat_init, 2));
try
    dist_type = 'sqEuclidean';

    if strcmpi(config.metric, 'cosine')
        rownorm = sqrt(sum(X_concat_init.^2, 2)) + 1e-12;
        X_concat_init = bsxfun(@rdivide, X_concat_init, rownorm);
        dist_type = 'cosine';
    end

    [y_init, ~] = lite_kmeans(X_concat_init, config.k, ...
                              'MaxIter', 100, ...
                              'Replicates', config.kmeans_reps, ...
                              'Distance', dist_type);
catch
    y_init = kmeans(X_concat_init, config.k, ...
                    'MaxIter', 100, ...
                    'Replicates', config.kmeans_reps, ...
                    'EmptyAction','drop');
end


Yv = cell(1,nV); Cv = cell(1,nV); K_diag_weights = cell(1, nV);
anchor_states = cell(1,nV); 
for v = 1:nV
    Yv{v} = one_hot(y_init, config.k);
    % Local orthogonal approximation (Eq. 7), distinct from global blocks.
    [Uc,~,Vc] = svd(full(Yv{v}' * Hv{v}), 'econ');
    Cv{v} = Uc * Vc';
    K_diag_weights{v} = ones(N, 1);
end
server.Y = one_hot(y_init, config.k);
% Initial global centroids in the concatenated latent space (N-by-Vk).
% The rectangular Procrustes factor satisfies C*C' = I_k.
Hc_init = cat(2,Hv{:});
[Uc,~,Vc] = svd(full(server.Y' * Hc_init), 'econ');
server.C = Uc * Vc';
server.C_blocks = cell(1,nV);
for v = 1:nV
    server.C_blocks{v} = server.C(:,(v-1)*config.k+(1:config.k));
end

%% ---------- Main Loop ----------
T = config.T;
logs.acc = []; logs.nmi = []; logs.ari = []; logs.purity = [];
logs.z_updates = cell(T,nV);
logs.h_updates = cell(T,nV);
logs.local_approx_updates = cell(T,nV);
logs.server_updates = cell(T,1);
best = struct('t',0,'nmi',-inf,'acc',0,'ari',0,'purity',0,'Y',server.Y,'assign',[]);
patience = 10; 
no_improve = 0;

for t = 1:T
    tt = tic;
    
    % Reconstruct feedback initially and after each server synchronization.
    % Anchor refresh has its own independent period inside each client.
    if t == 1 || (config.R_comm > 0 && mod(t-1,config.R_comm) == 0)
        % Eq. (18): retain compact labels/degrees, not Y*Y' or L.
        L_sum = consensus_laplacian_operator(server.Y);
    end

    % Client
    for v = 1:nV
        [Yv{v}, Cv{v}, Hv{v}, X_completed_list{v}, K_diag_weights{v}, anchor_states{v}] = ...
            client_step_RWC(X_incomplete_list{v}, Omega_list{v}, X_completed_list{v}, ...
                            L_sum, Hv{v}, Yv{v}, Cv{v}, ...
                            server.C_blocks{v}, K_diag_weights{v}, config, anchor_states{v});
        if isfield(anchor_states{v},'z_update') && ...
                anchor_states{v}.last_refresh == anchor_states{v}.iteration
            logs.z_updates{t,v} = anchor_states{v}.z_update;
        end
        if isfield(anchor_states{v},'h_update')
            logs.h_updates{t,v} = anchor_states{v}.h_update;
        end
        if isfield(anchor_states{v},'local_approx_update')
            logs.local_approx_updates{t,v} = anchor_states{v}.local_approx_update;
        end
    end

    % Server: preserve the paired hard labels and centroids from Eqs. 16-17.
    if config.R_comm > 0 && mod(t, config.R_comm) == 0
        server = server_step_plus(server, make_clients_from(Yv, Cv), config);
        logs.server_updates{t} = server.update_info;
        % No extra label permutation or temporal mixing: server.Y already
        % uses the cluster coordinates of server.C. Clients retain Yv/Cv;
        % only the separate C_blocks provide global guidance next round.
    end

    % Eval
    assign = argmax_rows(server.Y);
    assign = grp2idx(assign);
    metrics = evaluate_cluster_metrics(assign, gt_vec);
    
    logs.acc(end+1)=metrics.acc; logs.nmi(end+1)=metrics.nmi; 
    logs.ari(end+1)=metrics.ari; logs.purity(end+1)=metrics.purity;
    
    fprintf('Round %2d | ACC=%.4f  NMI=%.4f  ARI=%.4f | %.2fs\n', ...
        t, metrics.acc, metrics.nmi, metrics.ari, toc(tt));

    if metrics.nmi > best.nmi + 1e-5
        best.t = t; best.nmi = metrics.nmi; best.acc = metrics.acc; 
        best.ari = metrics.ari; best.purity = metrics.purity;
        best.assign = assign; best.Y = server.Y;
        no_improve = 0;
    else
        no_improve = no_improve + 1;
    end
    if t >= 20 && no_improve >= patience
        fprintf('Early stop at t=%d\n', t);
        break;
    end
end

R.final = struct('acc', logs.acc(end), 'nmi', logs.nmi(end), 'ari', logs.ari(end), 'purity', logs.purity(end));
R.best  = struct('acc', best.acc, 'nmi', best.nmi, 'ari', best.ari, 'purity', best.purity);
R.final.assign = assign;
R.best.assign = best.assign;
R.protocol.initial_assign = y_init;
R.protocol.missing_rows = cell(1,nV);
for v=1:nV
    R.protocol.missing_rows{v} = find(sum(Omega_list{v},2)==0);
end
R.protocol.structure_z_enabled = config.structure_z_enabled;
R.protocol.H_initialization = H_init_info;
R.protocol.h_gradient_mode = config.h_gradient_mode;
R.protocol.h_monotone_safeguard = config.h_monotone_safeguard;
R.protocol.completion_init_mode = config.completion_init_mode;
R.protocol.local_approx_mode = config.local_approx_mode;
R.protocol.server_consensus_mode = config.server_consensus_mode;
R.logs = logs;
end

% === Helpers ===
function gt_vec = normalize_gt(gt, strict)
    if nargin<2, strict=true; end
    if isempty(gt), gt_vec=[]; return; end
    if isvector(gt), gt_vec=gt(:); else, [~,gt_vec]=max(gt,[],2); end
    if min(gt_vec)==0, gt_vec=gt_vec+1; end
end
function cfg = set_defaults(cfg, df)
    fn=fieldnames(df); for i=1:numel(fn), if ~isfield(cfg,fn{i}), cfg.(fn{i})=df.(fn{i}); end; end
end
function idx = argmax_rows(Y), [~,idx]=max(Y,[],2); end
function clients = make_clients_from(Ycell, Ccell)
    for i=1:numel(Ycell), clients{i}.Y=Ycell{i}; clients{i}.C=Ccell{i}; end
end

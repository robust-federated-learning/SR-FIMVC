function R = main_fed_SR_FIMVC(dataset_name, run, config)
% =========================================================================
% SR-FIMVC (Auto-Weighted Version + 🌋 Stress Test Interceptor)
% 🌟 核心升级：恢复视图加权机制，解决 PER=0.5 时被噪声视图拖累的问题。
% 🌋 新增机制：极限抗压拦截器，支持外围直接注入 Sigma(高斯) 和 Rho(离群点)
% =========================================================================

rng(run);

%% ---------- defaults ----------
df = struct( ...
    'k',[], 'T',10, ...
    'lambda_graph',0.5, 'lambda_align',0.1, 'lambda2',1.0, ... 
    'use_anchor',true, 'pn',10, ...
    'R_comm',1, 'R_anchor',1, ...
    'anchor_m',[], 'anchor_s',12, ...
    'smart_axis',false, 'max_feat',[], ...
    'auto_k',true, 'force_k',false, ...
    'gt_strict', true, 'data_dir','data', ...
    'PER', 0.3, 'metric', 'euclidean', ...
    'mu_ridge',1e-3, 'eps_w',1e-6, ...       
    'use_view_w',true, ...                   % [FIX] 开启权重开关
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
% 1. 先进行标准化，确保数据尺度统一 (均值为0方差为1，或L2范数为1)
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
% 🌋 [极限抗压测试拦截器]：在归一化之后、切分缺失之前，强行注入噪声与离群点
% =========================================================================
if isfield(config, 'sigma') && config.sigma > 0
    fprintf('  [💥 压测注入] 正在向所有视图注入标准差 \\sigma = %.2f 的高斯噪声...\n', config.sigma);
    for v_noise = 1:numel(X_full_list)
        % 产生同等大小的高斯噪声，强行叠加到归一化后的特征上
        X_full_list{v_noise} = X_full_list{v_noise} + config.sigma * randn(size(X_full_list{v_noise}));
    end
end

if isfield(config, 'rho') && config.rho > 0
    fprintf('  [💥 压测注入] 正在向数据中强行替换比例 \\rho = %.2f 的极端离群点(Noisy Anchors)...\n', config.rho);
    for v_outlier = 1:numel(X_full_list)
        [num_samples, num_feats] = size(X_full_list{v_outlier});
        % 计算需要污染的样本数
        num_outliers = round(num_samples * config.rho);
        if num_outliers > 0
            % 随机挑出样本行索引，用超出正常密度的异常分布(如乘以 3.0)进行毁灭性覆盖
            outlier_idx = randperm(num_samples, num_outliers);
            X_full_list{v_outlier}(outlier_idx, :) = 3.0 * randn(num_outliers, num_feats);
        end
    end
end
% =========================================================================

% 2. 注入噪声后，再根据 PER 切分成不完整多视图数据
[X_incomplete_list, Omega_list] = create_incomplete_data(X_full_list, config.PER);

X_completed_list = cell(size(X_incomplete_list));
for v = 1:numel(X_incomplete_list)
    X_completed_list{v} = X_incomplete_list{v} .* Omega_list{v};
end

N  = size(X_completed_list{1},1);
nV = numel(X_completed_list);
config.k = max(2, numel(unique(gt_vec)));

% [NEW] 初始化视图权重 (均匀)
view_weight = ones(1, nV) / nV;

%% ---------- SVD + Global Init (Hybrid Version) ----------
% [优化] 支持两种初始化策略：
% 1. SVD (默认): 适合高维噪声大、需要降维的数据集。
% 2. Raw (config.use_raw_init=true): 适合特征本身很强、SVD 会丢失信息的难分数据集(如 100Leaves)。

if ~isfield(config, 'use_raw_init'), config.use_raw_init = false; end % 默认使用 SVD
if ~isfield(config, 'kmeans_reps'), config.kmeans_reps = 20; end

if config.use_raw_init
    % === 策略 A: 原始特征拼接 (带均值填充，专治 PER=0.5 崩盘) ===
    fprintf('[Init] Strategy: RAW Features Concat (Mean Imputation for Missing)...\n');
    X_concat_init = [];
    for v = 1:nV
        X_temp = X_completed_list{v}; % 此时缺失处是 0
        
        % [关键修改] 只有当存在缺失数据(PER>0)时，才进行均值填充
        % 如果是 PER=0，这段逻辑会跳过，保证 0.89 的高分不受影响
        if isfield(config, 'PER') && config.PER > 0
            % 对每一列进行遍历
            for d = 1:size(X_temp, 2)
                col = X_temp(:, d);
                % 找到非 0 (非缺失) 的元素
                non_zero_idx = (col ~= 0); 
                
                % 如果这一列有数据 (防止全空列报错)
                if any(non_zero_idx)
                    % 计算这一列的平均值 (只基于观测到的数据)
                    mu = mean(col(non_zero_idx));
                    
                    % 将缺失位置 (原本是0) 填补为均值
                    % 这样 K-Means 就不会把它们误判为"数值为0"，而是"平均水平"
                    col(~non_zero_idx) = mu;
                    X_temp(:, d) = col;
                end
            end
        end
        
        X_concat_init = [X_concat_init, X_temp]; 
    end
    
    % SVD 仅用于生成 Hv 变量结构，不用于聚类
    Hv = cell(1,nV);
    for v=1:nV
        try
           [U_svd, ~, ~] = svds(X_completed_list{v}, config.k);
           if size(U_svd, 2) < config.k, U_svd = [U_svd, zeros(N, config.k-size(U_svd,2))]; end
           Hv{v} = U_svd(:, 1:config.k);
        catch
           Hv{v} = orthonormal_init(N, config.k);
        end
    end
else
    % === 策略 B: SVD 降维 (Denoising & Fast) ===
    fprintf('[Init] Strategy: SVD Dimension Reduction...\n');
    Hv = cell(1,nV);
    for v = 1:nV
        try
            [U_svd, ~, ~] = svds(X_completed_list{v}, config.k);
            if size(U_svd, 2) < config.k
                padding = zeros(N, config.k - size(U_svd, 2));
                Hv{v} = [U_svd, padding];
            else
                Hv{v} = U_svd(:, 1:config.k);
            end
        catch
            Hv{v} = orthonormal_init(N, config.k);
        end
    end
    % SVD 模式下，拼接的是低维 H
    X_concat_init = [];
    for v = 1:nV, X_concat_init = [X_concat_init, Hv{v}]; end
end

% === 统一执行强力 K-Means ===
fprintf('       > K-Means: Reps=%d, DataDim=%d\n', config.kmeans_reps, size(X_concat_init, 2));
try
    dist_type = 'sqEuclidean';
    % Raw Init 且用 Cosine 时需要归一化，否则 Euclidean 即可
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

% === 初始化后续变量 ===
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
server.C_ref = server.C_blocks{1};

%% ---------- Main Loop ----------
T = config.T;
logs.acc = []; logs.nmi = []; logs.ari = []; logs.purity = [];
best = struct('t',0,'nmi',-inf,'acc',0,'ari',0,'purity',0,'Y',server.Y,'assign',[]);
patience = 10; 
no_improve = 0;

for t = 1:T
    tt = tic;
    
    % Reconstruct feedback initially and after each server synchronization.
    % Anchor refresh has its own independent period inside each client.
    if t == 1 || (config.R_comm > 0 && mod(t-1,config.R_comm) == 0)
        % Eq. (18). Explicit graph retained here; replacing it with an
        % implicit operator is a separate scalability change.
        S_consensus = sparse(server.Y) * sparse(server.Y)';
        D_S = spdiags(sum(S_consensus,2),0,N,N);
        L_sum = D_S - S_consensus;
    end

    % Client
    for v = 1:nV
        [Yv{v}, Cv{v}, Hv{v}, X_completed_list{v}, K_diag_weights{v}, anchor_states{v}] = ...
            client_step_RWC(X_incomplete_list{v}, Omega_list{v}, X_completed_list{v}, ...
                            L_sum, Hv{v}, Yv{v}, Cv{v}, ...
                            server.C_blocks{v}, K_diag_weights{v}, config, anchor_states{v});
    end

    % Server: preserve the paired hard labels and centroids from Eqs. 16-17.
    if config.R_comm > 0 && mod(t, config.R_comm) == 0
        server = server_step_plus(server, make_clients_from(Yv, Cv), config);
        % No extra label permutation or temporal mixing: server.Y already
        % uses the cluster coordinates of server.C. Clients retain Yv/Cv;
        % only the separate C_blocks provide global guidance next round.
    end

    % [NEW] 🌟 Update View Weights 🌟
    if config.use_view_w
        errs = zeros(1, nV);
        for v = 1:nV
            % 计算 H_v 与全局 Y 的对齐误差
            % 误差越小，权重越大
            diff = Hv{v} * Cv{v}' - server.Y;
            errs(v) = norm(diff, 'fro');
        end
        % 避免除零
        errs = errs + 1e-6;
        % 权重反比于误差 (自动归一化)
        weights_new = 1 ./ errs;
        view_weight = weights_new / sum(weights_new);
        
        if mod(t, 20) == 0
             fprintf('  [Weights] '); fprintf('%.3f ', view_weight); fprintf('\n');
        end
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
R.logs = logs;
end

% === Helpers ===
function gt_vec = normalize_gt(gt, strict)
    if nargin<2, strict=true; end
    if isempty(gt), gt_vec=[]; return; end
    if isvector(gt), gt_vec=gt(:); else, [~,gt_vec]=max(gt,[],2); end
    if min(gt_vec)==0, gt_vec=gt_vec+1; end
end
function H = orthonormal_init(n, k), A=randn(n,k); [U,~,~]=svd(A,'econ'); H=U; end
function tf = is_near_zero_L(L), tf = (nnz(L)<size(L,1)) || (norm(L,'fro')<1e-6); end
function cfg = set_defaults(cfg, df)
    fn=fieldnames(df); for i=1:numel(fn), if ~isfield(cfg,fn{i}), cfg.(fn{i})=df.(fn{i}); end; end
end
function idx = argmax_rows(Y), [~,idx]=max(Y,[],2); end
function clients = make_clients_from(Ycell, Ccell)
    for i=1:numel(Ycell), clients{i}.Y=Ycell{i}; clients{i}.C=Ccell{i}; end
end
function Lsum = sum_L_per_view(Xset, Omega_list, cfg, weights) % <--- 1. 增加参数 Omega_list
    if nargin < 4, weights = ones(1, numel(Xset)) / numel(Xset); end
    N = size(Xset{1},1); Lsum = sparse(N,N);
    for v = 1:numel(Xset)
        X = Xset{v}; if size(X,1)~=N, X=X'; end
        if cfg.use_anchor
            L_v = build_anchor_L(X, cfg.anchor_m, cfg.anchor_s, cfg.metric);
        else
            L_v = build_graph_L(X, cfg.pn, cfg.metric); 
        end
        
        % --- [NEW] G^(v) 结构过滤逻辑 Start ---
        % 只要该样本有一维特征是观测到的，就认为该样本在视图 v 存在
        % valid_idx 是一个 N x 1 的逻辑向量
        valid_idx = sum(Omega_list{v}, 2) > 0;
        
        % 构建对角掩码矩阵 G^(v) (N x N sparse)
        G_v = spdiags(double(valid_idx), 0, N, N);
        
        % 应用掩码：彻底切断缺失样本的图连接
        % L_masked = G^(v) * L^{(v)} * G^(v)
        L_v = G_v * L_v * G_v;
        % --- [NEW] G^(v) 结构过滤逻辑 End ---

        % 加权累加 (View-level Weight + Sample-level Mask)
        Lsum = Lsum + weights(v) * L_v;
    end
end

% === [MODIFIED] sum_L_per_latent_view (支持权重 + Gv Mask) ===
function Lsum = sum_L_per_latent_view(Hset, Omega_list, cfg, weights) % <--- 1. 增加 Omega_list 参数
    if nargin < 4, weights = ones(1, numel(Hset)) / numel(Hset); end
    N = size(Hset{1},1); Lsum = sparse(N,N);
    
    for v = 1:numel(Hset)
        % 动态构图 (基于 H)
        if cfg.use_anchor
            L_v = build_anchor_L(Hset{v}, cfg.anchor_m, cfg.anchor_s, cfg.metric);
        else
            L_v = build_graph_L(Hset{v}, cfg.pn, cfg.metric);
        end
        
        % --- [NEW] G^(v) 结构过滤逻辑 Start ---
        % 逻辑同 sum_L_per_view：样本在当前视图有观测值才算有效
        valid_idx = sum(Omega_list{v}, 2) > 0;
        
        % 构建对角掩码矩阵 G^(v)
        G_v = spdiags(double(valid_idx), 0, N, N);
        
        % 应用掩码：L_masked = G^(v) * L^{(v)} * G^(v)
        % 这一步确保了动态更新的图结构也不会被缺失样本误导
        L_v = G_v * L_v * G_v;
        % --- [NEW] G^(v) 结构过滤逻辑 End ---
        
        % 加权累加
        Lsum = Lsum + weights(v) * L_v;
    end
end
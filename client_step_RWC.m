function [Y_v, C_v, H_v, X_completed, K_diag_weights] = ...
    client_step_RWC(X_incomplete, Omega, X_completed_prev, ...
                    L_sum, H_v_prev, Y_v_prev, C_v_prev, ...
                    C_server_v, K_diag_weights_prev, ...
                    config)

% 获取参数
k = config.k;
N = size(X_incomplete, 1);
eps_w = config.eps_w; 
mu_ridge = config.mu_ridge;
lambda_graph = config.lambda_graph;
lambda_align = config.lambda_align;

% 检查是否存在缺失数据
has_missing = any(Omega(:) == 0);

if has_missing
    %% ================== 分支 A: 缺失处理逻辑 (对应论文 Eq. 3-12) ==================
    % 仅在存在缺失时启用动态锚点 Z, U 和对齐 A 
    
    m = config.anchor_m; 
    if isempty(m), m = min(round(N * 0.1), 50); end

    % 1. 动态锚点指派 (Z_v) - 捕捉局部流形 
    [Z_idx, U_v_temp] = lite_kmeans(X_completed_prev, m, 'MaxIter', 3);
    Z_v = sparse(1:N, Z_idx, 1, N, m);

    % 2. 结构对齐矩阵 (A_v) - 对应 Eq. 12 
    valid_idx = sum(Omega, 2) > 0;
    G_v = spdiags(double(valid_idx), 0, N, N);
    B_v = Z_v' * G_v * L_sum * G_v' * Z_v;
    S_mat = Z_v' * Z_v + 1e-6 * eye(m);
    
    % ==================== [防爆特征值求解 替换开始] ====================
    M_left = full((B_v + B_v')/2 + 1e-4*eye(m));
    M_right = full(S_mat);

    if m <= k + 2
        % 当特征维度 m 小于或等于类别数 k 时，eigs 会崩溃
        % 切换到底层 eig 求解全量特征值
        [V_eig, D_eig] = eig(M_left, M_right);
        eigenvalues = diag(D_eig);
        [~, sort_idx] = sort(real(eigenvalues), 'ascend');
        
        take_num = min(m, k);
        A_v = V_eig(:, sort_idx(1:take_num));
        
        % 维度不足部分用全 0 补齐，维持 A_v 的 [m x k] 标准形状不变
        if take_num < k
            A_v = [A_v, zeros(m, k - take_num)];
        end
    else
        % 维度充足时，优先使用 eigs 加速计算
        try
            [A_v, ~] = eigs(M_left, M_right, k, 'smallestreal');
        catch
            % 终极兜底：防止因极端缺失导致矩阵奇异、eigs 不收敛
            [V_eig, D_eig] = eig(M_left, M_right);
            [~, sort_idx] = sort(real(diag(D_eig)), 'ascend');
            A_v = V_eig(:, sort_idx(1:k));
        end
    end
    % ==================== [防爆特征值求解 替换结束] ====================

    % 3. 残差权重 (w) - 对应 Eq. 4 (RWC 机制) 
    X_rec = Z_v * U_v_temp;
    residuals_sq = sum(((X_incomplete - X_rec) .* Omega).^2, 2);
    w = 1.0 ./ sqrt(residuals_sq + eps_w); 
    
    % 处理极端缺失样本
    is_missing = sum(Omega, 2) == 0;
    if any(is_missing)
        w(is_missing) = median(w(~is_missing)); 
    end
    W_sq = sparse(1:N, 1:N, w.^2, N, N); 

    % 4. 锚点基 (U_v) - 对应 Eq. 11 
    U_v = (Z_v' * W_sq * Z_v + mu_ridge * eye(m)) \ (Z_v' * W_sq * X_completed_prev);

    % 5. 特征补全 (X_com) - 对应 Eq. 10 
    X_rec_final = Z_v * U_v;
    P_v = (1 - Omega) .* X_rec_final;
    X_completed = (Omega .* X_incomplete) + P_v;

else
    %% ================== 分支 B: 完整数据逻辑 (退化模式) ==================
    % 当数据完整时，跳过锚点重构，避免近似误差
    % 此时 w 恒为 1，X_completed 即为原始特征
    
    X_completed = X_incomplete;
    w = ones(N, 1);
    K_diag_weights = w;
end

K_diag_weights = w; 

%% ---------- 步骤 4/5: 隐式表示学习 H_v (对应论文 Eq. 13) ----------
% 无论是否有缺失，这一步都是核心，利用结合律实现 O(N) 
H_weighted = bsxfun(@times, H_v_prev, w.^2); 

% 1. 隐式核项: X * (X' * H)
Term1_K = X_completed * (X_completed' * H_weighted);

% 2. 全局结构反馈项: L_sum * H
Term1_L = lambda_graph * (L_sum * H_weighted);

% 3. 共识对齐项
Term2 = lambda_align * full(Y_v_prev) * C_server_v;

% 计算代理梯度并进行正交投影 
M = (Term1_K - Term1_L) + Term2;
[U, ~, V] = svd(M, 'econ');
H_v = U * V'; 

%% ---------- 步骤 6: 离散近似 (Y_v, C_v) (对应论文 Eq. 14) ----------
[Uc, ~, Vc] = svd(full(Y_v_prev)' * H_v, 'econ');
C_v = Uc * Vc';

[~, Y_idx] = max(H_v * C_v', [], 2);
Y_v = full(sparse(1:N, Y_idx, 1, N, k));

end
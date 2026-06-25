function L = fallback_knn_L(Xlist_or_X, k)
% Fallback: 构建一个 N×N 的 kNN 拉普拉斯，作为“图近零时”的兜底
% [极速稀疏防爆版] - 彻底消灭所有 N*N 稠密矩阵，永不爆内存

    if nargin < 2 || isempty(k), k = 10; end

    % --- 1. 拼出 N×d 的样本表示（保持原逻辑不变） ---
    if iscell(Xlist_or_X)
        N = []; Xcat = [];
        for v = 1:numel(Xlist_or_X)
            Xv = double(Xlist_or_X{v});
            if size(Xv,1) < size(Xv,2)   % 常见 d×N
                Xv = Xv';
            end
            if isempty(N), N = size(Xv,1); end
            if size(Xv,1) ~= N
                error('fallback_knn_L: 视图 %d 的样本数与前一视图不一致。', v);
            end
            Xcat = [Xcat, zscore(Xv)];
        end
        X = Xcat;
    else
        X = double(Xlist_or_X);
        if size(X,1) < size(X,2)
            X = X';
        end
        X = zscore(X);
    end
    N = size(X,1);

    % --- 2. 极速 KNN 搜索 (拆除 pdist 距离矩阵炸弹) ---
    kk = min(k, N-1);
    try
        % 优先使用 knnsearch (C++底层，极度省内存)
        [idx, ~] = knnsearch(X, X, 'K', kk + 1, 'Distance', 'euclidean');
    catch
        % 兜底：老版本 MATLAB 用 pdist2 取最小的 k 个
        [~, idx] = pdist2(X, X, 'euclidean', 'Smallest', kk+1);
        idx = idx'; % 统一转置为 N x (kk+1)
    end

    % --- 3. 构造纯稀疏图 W (拆除 zeros(N,N) 炸弹) ---
    row_idx = zeros(N * kk, 1);
    col_idx = zeros(N * kk, 1);
    val_W   = ones(N * kk, 1);

    count = 0;
    for i = 1:N
        neighbors = idx(i, 2:kk+1); % 跳过自己
        for j = 1:kk
            count = count + 1;
            row_idx(count) = i;
            col_idx(count) = neighbors(j);
            % val_W 在初始化时已经是 1 了
        end
    end

    W = sparse(row_idx(1:count), col_idx(1:count), val_W(1:count), N, N);
    W = max(W, W'); % 稀疏矩阵对称化

    % --- 4. 拉普拉斯 (拆除 diag(d) 炸弹) ---
    d = sum(W, 2);
    D_mat = spdiags(d, 0, N, N); % 使用稀疏对角阵
    L = D_mat - W;               % 全程纯稀疏计算
end
function L = build_graph_L(X, k, metric)
% BUILD_GRAPH_L (PKN Version): 概率自适应近邻图构建 (极速稀疏优化版)
% 确保使用 (d_k+1 - d)/(...) 公式，完美防内存溢出

    if nargin < 2 || isempty(k), k = 10; end
    if nargin < 3 || isempty(metric), metric = 'euclidean'; end
    
    [N, ~] = size(X);

    % ---- 1. 高效近邻搜索 (彻底拆除 N*N 距离矩阵炸弹) ----
    try
        % 优先使用 MATLAB 内置的 knnsearch (底层C++优化，极低内存占用)
        if strcmpi(metric, 'cosine')
            [idx, D] = knnsearch(X, X, 'K', k+2, 'Distance', 'cosine');
        else
            [idx, D] = knnsearch(X, X, 'K', k+2, 'Distance', 'euclidean');
        end
    catch
        % 安全兜底: 如果老版本没有机器学习工具箱，回退到普通计算并及时清理内存
        X_in = X'; 
        if strcmpi(metric, 'cosine')
            normX = sqrt(sum(X_in.^2, 1)) + 1e-12;
            X_in = bsxfun(@rdivide, X_in, normX);
        end
        aa = sum(X_in.*X_in); 
        D_full = bsxfun(@plus, aa', aa) - 2*(X_in'*X_in);
        D_full(D_full<0) = 0;
        D_full = sqrt(D_full); 
        [D_sorted, idx_sorted] = sort(D_full, 2);
        
        % 只保留前 k+2 个，防止爆内存
        D = D_sorted(:, 1:k+2);
        idx = idx_sorted(:, 1:k+2);
        clear D_full D_sorted idx_sorted; % 算完立刻释放大矩阵
    end

    % ---- 2. PKN 权重计算 (彻底拆除 W = zeros(N,N) 炸弹) ----
    % 用三元组记录有效坐标，最后生成纯稀疏矩阵
    row_idx = zeros(N * k, 1);
    col_idx = zeros(N * k, 1);
    val_W   = zeros(N * k, 1);

    count = 0;
    for i = 1:N
        % 取第 2 到 k+2 个近邻（跳过自己，自己是第1个）
        id = idx(i, 2:k+2);      
        di = D(i, 2:k+2);        
        
        dist_k_plus_1 = di(k+1); 
        
        % 计算 PKN 自适应权重 (完美匹配论文公式)
        numerator = dist_k_plus_1 - di(1:k); 
        denominator = k * dist_k_plus_1 - sum(di(1:k)) + eps;
        
        w_row = numerator / denominator;
        w_row = max(0, w_row); % 确保非负
        
        % 写入三元组
        for j = 1:k
            count = count + 1;
            row_idx(count) = i;
            col_idx(count) = id(j);
            val_W(count)   = w_row(j);
        end
    end

    % 一次性生成完美的纯稀疏矩阵 W
    W = sparse(row_idx(1:count), col_idx(1:count), val_W(1:count), N, N);

    % ---- 3. 计算归一化拉普拉斯 (全程保持稀疏状态) ----
    W = (W + W') / 2; 
    d_sum = sum(W, 2); 
    d_sum(d_sum < 1e-12) = 1e-12;
    
    Dm12 = spdiags(1 ./ sqrt(d_sum), 0, N, N); % 稀疏度矩阵
    L = speye(N) - Dm12 * W * Dm12;            % 稀疏单位阵 - 稀疏传播阵
end
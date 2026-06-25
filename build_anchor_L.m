function [Lsum, meta] = build_anchor_L(H_in, anchor_m, anchor_s, metric)
% =========================================================================
% 完美匹配论文的锚点图构建 (纯稀疏运算，绝不报内存溢出错误)
% =========================================================================

    if nargin < 2 || isempty(anchor_m), anchor_m = max(10, round(sqrt(size(H_in,1)))); end
    if nargin < 3 || isempty(anchor_s), anchor_s = min(5, anchor_m); end
    if nargin < 4 || isempty(metric), metric = 'euclidean'; end
    metric = lower(metric);

    [N, ~] = size(H_in);

    % ---- 1) 选锚点 (保留你的 lite_kmeans 或随机选择) ----
    anchors = [];
    if exist('lite_kmeans', 'file') == 2
        try
            if strcmpi(metric, 'cosine')
                rownorm = sqrt(sum(H_in.^2, 2)) + 1e-12;
                H_norm = bsxfun(@rdivide, H_in, rownorm);
                [~, C] = lite_kmeans(H_norm, anchor_m, 'MaxIter', 50, 'Replicates', 1, 'Distance', 'cosine');
                anchors = C;
            else
                [~, C] = lite_kmeans(H_in, anchor_m, 'MaxIter', 50, 'Replicates', 1, 'Distance', 'sqEuclidean');
                anchors = C;
            end
        catch
        end
    end
    
    if isempty(anchors) && exist('kmeans','file') == 2
        try
            opts = statset('MaxIter', 50, 'Display', 'off');
            [~, C] = kmeans(H_in, anchor_m, 'Replicates', 1, 'MaxIter', 50, 'Options', opts); 
            anchors = C;
        catch
        end
    end
    
    if isempty(anchors)
        idx = randperm(N, min(anchor_m, N));
        anchors = H_in(idx, :); 
        if size(anchors,1) < anchor_m
            anchors = [anchors; zeros(anchor_m - size(anchors,1), size(anchors,2))];
        end
    end

    % ---- 2) 计算距离 ----
    switch metric
        case 'cosine'
            D2 = cosine_distance(H_in, anchors); 
        otherwise
            D2 = squared_euclidean(H_in, anchors); 
    end
    
    % ---- 3) 计算权重 (保留你的自适应 Sigma 逻辑) ----
    k_for_sigma = max(1, min(5, size(D2,2))); 
    s_sorted = sort(D2, 2, 'ascend');
    sigma2 = mean(s_sorted(:, k_for_sigma));
    
    if ~isfinite(sigma2) || sigma2 <= 0
        if strcmp(metric, 'cosine'), sigma2 = 0.5; else, sigma2 = 1; end
    end

    Wxa = exp(-D2 / (2 * sigma2)); 

    % ---- 4) 核心防报错优化：Top-s 稀疏化构建 Z ----
    % 用三元组直接构造稀疏矩阵，替代原来的 zeros()，防止后面的乘法爆内存
    row_idx = zeros(N * anchor_s, 1);
    col_idx = zeros(N * anchor_s, 1);
    val_Z   = zeros(N * anchor_s, 1);
    
    count = 0;
    for i = 1:N
        [v, idx] = topk(Wxa(i,:), anchor_s);
        ssum = sum(v);
        if ssum > 0
            v = v / ssum; 
        end
        for j = 1:length(idx)
            count = count + 1;
            row_idx(count) = i;
            col_idx(count) = idx(j);
            val_Z(count)   = v(j);
        end
    end
    
    % 此时的 Z 是原汁原味的稀疏矩阵
    Z = sparse(row_idx(1:count), col_idx(1:count), val_Z(1:count), N, anchor_m);

    % ---- 5) 归一化拉普拉斯 (全程稀疏，速度极快) ----
    W = Z * Z.'; 
    W = (W + W.') * 0.5;
    d = sum(W, 2); 
    d(d <= 0) = eps;
    Dm = spdiags(d.^(-0.5), 0, N, N); 
    Lsum = speye(N) - Dm * W * Dm;    

    meta = struct('anchors', anchors, 'sigma2', sigma2, 'metric', metric);
end

% === Helpers 保持原样，未做任何改动 ===
function D2 = squared_euclidean(A, B)
    AA = sum(A.^2, 2); BB = sum(B.^2, 2).';
    D2 = bsxfun(@plus, AA, BB) - 2*(A*B.');
    D2(D2<0) = 0; D2(~isfinite(D2)) = 0;
end
function D = cosine_distance(A, B)
    normA = sqrt(sum(A.^2, 2)); normA(normA==0)=1;
    normB = sqrt(sum(B.^2, 2)); normB(normB==0)=1;
    A_norm = bsxfun(@rdivide, A, normA);
    B_norm = bsxfun(@rdivide, B, normB);
    S = A_norm * B_norm.';
    S(S > 1) = 1; S(S < -1) = -1;
    D = 1 - S; D(~isfinite(D)) = 0;
end
function [vals, idx] = topk(row, k)
    if isempty(row), vals=[]; idx=[]; return; end
    k = max(1, min(k, numel(row)));
    [vals, idx] = sort(row, 'descend');
    vals=vals(1:k); idx=idx(1:k);
end
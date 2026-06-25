function metrics = evaluate_cluster_metrics(assign, gt)
% Robust metrics: ACC, NMI, Purity, F1, ARI
% 修复：使用标准的列联表组合计数法计算 ARI，以解决数值溢出和归一化为 0 的问题。

    % --- sanitize labels ---
    assign = assign(:); gt = gt(:);
    n = min(numel(assign), numel(gt));
    assign = assign(1:n); gt = gt(1:n);
    
    % --- ACC via Hungarian (with safe fallback) ---
    acc = clustering_acc_safe(gt, assign);

    % --- NMI ---
    nmi = nmi_safe(gt, assign);

    % --- Purity ---
    purity = purity_safe(gt, assign);

    % --- F1 & ARI (pair-counting, with eps guards) ---
    [f1, ari] = f1_ari_safe(gt, assign); % <-- 最终修正后的 ARI/F1 计算

    metrics = struct('acc',acc,'nmi',nmi,'purity',purity,'f1',f1,'ari',ari);
end

function acc = clustering_acc_safe(y, yhat)
    y = y(:); yhat = yhat(:);
    c = unique(y); k = unique(yhat);
    cm = zeros(numel(c), numel(k));
    for i=1:numel(c)
        for j=1:numel(k)
            cm(i,j) = sum(y==c(i) & yhat==k(j));
        end
    end
    cost = max(cm(:)) - cm;
    acc = nan;
    % 假设您有 munkres/hungarian 函数
    if exist('munkres','file')==2
        [~, col] = munkres(cost);
        acc = sum(cm(sub2ind(size(cm),1:numel(c),col)));
    elseif exist('hungarian','file')==2
        col = hungarian(cost);
        acc = sum(cm(sub2ind(size(cm),1:numel(c),col)));
    else
        % greedy fallback
        acc = sum(max(cm,[],2));
    end
    acc = acc / numel(y);
end

function nmi = nmi_safe(y, yhat)
    y = y(:); yhat = yhat(:); N = numel(y);
    cy = unique(y); ch = unique(yhat);
    
    % --- 1. 计算联合概率 P(Y, C) 和互信息 MI ---
    MI = 0;
    for i=1:numel(cy)
        pi = sum(y==cy(i))/N;
        for j=1:numel(ch)
            pj = sum(yhat==ch(j))/N;
            pij = sum(y==cy(i) & yhat==ch(j))/N;
            
            if pij > 1e-10 
                denominator = pi * pj;
                if denominator < 1e-16  
                   denominator = 1e-16; 
                end
                MI = MI + pij * log(pij / denominator);
            end
        end
    end
    
    % --- 2. 计算熵 H1, H2 ---
    entropy_calc = @(p) -p .* log(p);
    
    % H1 (Entropy of y)
    py = arrayfun(@(v) sum(y==v)/N, cy);
    py(py <= 0) = 1; 
    H1 = sum(entropy_calc(py));
    
    % H2 (Entropy of yhat)
    ph = arrayfun(@(v) sum(yhat==v)/N, ch);
    ph(ph <= 0) = 1; 
    H2 = sum(entropy_calc(ph));

    % --- 3. 归一化 ---
    denominator = sqrt(H1 * H2);
    if denominator < 1e-10
        nmi = 0;
    else
        nmi = MI / denominator;
    end
end

function purity = purity_safe(y, yhat)
    cl = unique(yhat); N = numel(y);
    s = 0;
    for j=1:numel(cl)
        idx = (yhat==cl(j)); if ~any(idx), continue; end
        g = y(idx);
        vals = unique(y);
        cnt = arrayfun(@(v) sum(g==v), vals);
        s = s + max(cnt);
    end
    purity = s / N;
end

function [f1, ari] = f1_ari_safe(y, yhat)
    y = y(:); yhat = yhat(:); n = numel(y);
    
    % ------------------------ 核心优化：列联表 (Contingency Table) ------------------------
    % 仅生成 C x K 的小矩阵，彻底消灭 N x N 大矩阵，永不爆内存！
    cy = unique(y); ch = unique(yhat);
    C = zeros(numel(cy), numel(ch));
    for i = 1:numel(cy)
        for j = 1:numel(ch)
            C(i, j) = sum(y == cy(i) & yhat == ch(j));
        end
    end
    
    % 边缘和
    n_i = sum(C, 2); % 真实标签 (行和)
    m_j = sum(C, 1); % 聚类标签 (列和)
    
    % 组合计数函数 nchoosek(k, 2)
    nchoosek_2 = @(k) k .* (k - 1) / 2; 

    % --- 1. 计算 F1 Score (利用列联表直接推导，极速!) ---
    TP = sum(nchoosek_2(C(:)));       % 真实为同类且预测为同类的对数
    PP = sum(nchoosek_2(m_j(:)));     % 预测为同类的所有对数 (TP + FP)
    AP = sum(nchoosek_2(n_i(:)));     % 真实为同类的所有对数 (TP + FN)
    
    precision = TP / max(PP, eps);
    recall    = TP / max(AP, eps);
    f1 = 2 * precision * recall / max(precision + recall, eps);

    % --- 2. 计算 ARI (复用上面的组合数，一气呵成!) ---
    sum_n_i_pairs = AP; % 刚好等于 AP
    sum_m_j_pairs = PP; % 刚好等于 PP
    
    N_pairs = max(1, nchoosek_2(n)); 
    E_RI_numerator = (sum_n_i_pairs * sum_m_j_pairs) / N_pairs; 
    Max_RI_numerator = 0.5 * (sum_n_i_pairs + sum_m_j_pairs);

    % TP 其实就是 Rand Index (RI) 的分子
    numerator   = TP - E_RI_numerator;
    denominator = Max_RI_numerator - E_RI_numerator;
    
    if abs(denominator) < 1e-10 
        ari = 0; 
    else
        ari = numerator / denominator;
    end
end
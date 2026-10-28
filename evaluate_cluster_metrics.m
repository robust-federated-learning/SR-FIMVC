function metrics = evaluate_cluster_metrics(assign, gt)
% Robust metrics: ACC, NMI, Purity, F1, ARI


    % --- sanitize labels ---
    assign = assign(:); gt = gt(:);
    validateattributes(assign, {'numeric','logical'}, {'real','finite','nonempty'});
    validateattributes(gt, {'numeric','logical'}, {'real','finite','nonempty'});
    if numel(assign) ~= numel(gt)
        error('SRFIMVC:LabelLengthMismatch', ...
              'Predicted and ground-truth labels must have equal lengths.');
    end
    
    % --- ACC via exact one-to-one assignment (no external dependency) ---
    acc = clustering_acc_safe(gt, assign);

    % --- NMI ---
    nmi = nmi_safe(gt, assign);

    % --- Purity ---
    purity = purity_safe(gt, assign);

    % --- F1 & ARI (pair-counting, with eps guards) ---
    [f1, ari] = f1_ari_safe(gt, assign);

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
    % Zero padding represents unmatched classes/clusters when counts differ.
    n = max(size(cm));
    counts = zeros(n,n);
    counts(1:size(cm,1),1:size(cm,2)) = cm;
    cost = max(counts(:)) - counts;
    cols = exact_assignment(cost);
    matched = counts(sub2ind([n,n], (1:n)', cols));
    acc = sum(matched) / numel(y);
end

function cols = exact_assignment(cost)
% Shortest augmenting-path Hungarian algorithm, O(n^3).
% Column slot 1 is a dummy; slots 2:n+1 represent real columns.
    n = size(cost,1);
    u = zeros(n,1);
    v = zeros(n+1,1);
    p = zeros(n+1,1);
    way = zeros(n+1,1);
    for row = 1:n
        p(1) = row;
        j0 = 1;
        minv = inf(n+1,1);
        used = false(n+1,1);
        while true
            used(j0) = true;
            i0 = p(j0);
            delta = inf;
            j1 = 0;
            for j = 2:n+1
                if ~used(j)
                    cur = cost(i0,j-1) - u(i0) - v(j);
                    if cur < minv(j)
                        minv(j) = cur;
                        way(j) = j0;
                    end
                    if minv(j) < delta
                        delta = minv(j);
                        j1 = j;
                    end
                end
            end
            if ~isfinite(delta) || j1 == 0
                error('SRFIMVC:AssignmentFailed', ...
                      'Exact label assignment failed.');
            end
            for j = 1:n+1
                if used(j)
                    u(p(j)) = u(p(j)) + delta;
                    v(j) = v(j) - delta;
                else
                    minv(j) = minv(j) - delta;
                end
            end
            j0 = j1;
            if p(j0) == 0
                break;
            end
        end
        while true
            j1 = way(j0);
            p(j0) = p(j1);
            j0 = j1;
            if j0 == 1
                break;
            end
        end
    end
    cols = zeros(n,1);
    for j = 2:n+1
        cols(p(j)) = j-1;
    end
end

function nmi = nmi_safe(y, yhat)
    y = y(:); yhat = yhat(:); N = numel(y);
    cy = unique(y); ch = unique(yhat);
    

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
    

    entropy_calc = @(p) -p .* log(p);
    
    % H1 (Entropy of y)
    py = arrayfun(@(v) sum(y==v)/N, cy);
    py(py <= 0) = 1; 
    H1 = sum(entropy_calc(py));
    
    % H2 (Entropy of yhat)
    ph = arrayfun(@(v) sum(yhat==v)/N, ch);
    ph(ph <= 0) = 1; 
    H2 = sum(entropy_calc(ph));


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
    


    cy = unique(y); ch = unique(yhat);
    C = zeros(numel(cy), numel(ch));
    for i = 1:numel(cy)
        for j = 1:numel(ch)
            C(i, j) = sum(y == cy(i) & yhat == ch(j));
        end
    end
    

    n_i = sum(C, 2);
    m_j = sum(C, 1);
    

    nchoosek_2 = @(k) k .* (k - 1) / 2; 


    TP = sum(nchoosek_2(C(:)));
    PP = sum(nchoosek_2(m_j(:)));
    AP = sum(nchoosek_2(n_i(:)));
    
    precision = TP / max(PP, eps);
    recall    = TP / max(AP, eps);
    f1 = 2 * precision * recall / max(precision + recall, eps);


    sum_n_i_pairs = AP;
    sum_m_j_pairs = PP;
    
    N_pairs = max(1, nchoosek_2(n)); 
    E_RI_numerator = (sum_n_i_pairs * sum_m_j_pairs) / N_pairs; 
    Max_RI_numerator = 0.5 * (sum_n_i_pairs + sum_m_j_pairs);


    numerator   = TP - E_RI_numerator;
    denominator = Max_RI_numerator - E_RI_numerator;
    
    if abs(denominator) < 1e-10 
        ari = 0; 
    else
        ari = numerator / denominator;
    end
end
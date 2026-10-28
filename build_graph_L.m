function L = build_graph_L(X, k, metric)



    if nargin < 2 || isempty(k), k = 10; end
    if nargin < 3 || isempty(metric), metric = 'euclidean'; end
    
    [N, ~] = size(X);


    try

        if strcmpi(metric, 'cosine')
            [idx, D] = knnsearch(X, X, 'K', k+2, 'Distance', 'cosine');
        else
            [idx, D] = knnsearch(X, X, 'K', k+2, 'Distance', 'euclidean');
        end
    catch

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
        

        D = D_sorted(:, 1:k+2);
        idx = idx_sorted(:, 1:k+2);
        clear D_full D_sorted idx_sorted;
    end



    row_idx = zeros(N * k, 1);
    col_idx = zeros(N * k, 1);
    val_W   = zeros(N * k, 1);

    count = 0;
    for i = 1:N

        id = idx(i, 2:k+2);      
        di = D(i, 2:k+2);        
        
        dist_k_plus_1 = di(k+1); 
        

        numerator = dist_k_plus_1 - di(1:k); 
        denominator = k * dist_k_plus_1 - sum(di(1:k)) + eps;
        
        w_row = numerator / denominator;
        w_row = max(0, w_row);
        

        for j = 1:k
            count = count + 1;
            row_idx(count) = i;
            col_idx(count) = id(j);
            val_W(count)   = w_row(j);
        end
    end


    W = sparse(row_idx(1:count), col_idx(1:count), val_W(1:count), N, N);


    W = (W + W') / 2; 
    d_sum = sum(W, 2); 
    d_sum(d_sum < 1e-12) = 1e-12;
    
    Dm12 = spdiags(1 ./ sqrt(d_sum), 0, N, N);
    L = speye(N) - Dm12 * W * Dm12;
end
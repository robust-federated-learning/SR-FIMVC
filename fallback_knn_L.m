function L = fallback_knn_L(Xlist_or_X, k)



    if nargin < 2 || isempty(k), k = 10; end


    if iscell(Xlist_or_X)
        N = []; Xcat = [];
        for v = 1:numel(Xlist_or_X)
            Xv = double(Xlist_or_X{v});
            if size(Xv,1) < size(Xv,2)
                Xv = Xv';
            end
            if isempty(N), N = size(Xv,1); end
            if size(Xv,1) ~= N
                error('fallback_knn_L: View %d has an inconsistent sample count.', v);
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


    kk = min(k, N-1);
    try

        [idx, ~] = knnsearch(X, X, 'K', kk + 1, 'Distance', 'euclidean');
    catch

        [~, idx] = pdist2(X, X, 'euclidean', 'Smallest', kk+1);
        idx = idx';
    end


    row_idx = zeros(N * kk, 1);
    col_idx = zeros(N * kk, 1);
    val_W   = ones(N * kk, 1);

    count = 0;
    for i = 1:N
        neighbors = idx(i, 2:kk+1);
        for j = 1:kk
            count = count + 1;
            row_idx(count) = i;
            col_idx(count) = neighbors(j);

        end
    end

    W = sparse(row_idx(1:count), col_idx(1:count), val_W(1:count), N, N);
    W = max(W, W');


    d = sum(W, 2);
    D_mat = spdiags(d, 0, N, N);
    L = D_mat - W;
end

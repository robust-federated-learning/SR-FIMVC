function [H, H_cluster, info] = initialize_latent_view(X, k)
%INITIALIZE_LATENT_VIEW Build a feasible N-by-k latent initialization.
% H satisfies H'*H = I whenever N >= k. H_cluster contains only the
% data-supported left singular directions and is intended for initial
% clustering; its remaining columns are zero so that the deterministic
% orthogonal completion cannot inject artificial clustering information.

    if ~isnumeric(X) || ~isreal(X) || any(~isfinite(X(:)))
        error('initialize_latent_view:InvalidX', ...
              'X must be a finite real numeric matrix.');
    end
    [n, d] = size(X);
    if ~isscalar(k) || k ~= floor(k) || k < 1 || n < k
        error('initialize_latent_view:InvalidK', ...
              'k must be a positive integer no larger than size(X,1).');
    end

    q = min([k, n, d]);
    if q == 0
        U = zeros(n, 0);
        singular = zeros(0, 1);
    elseif min(n, d) <= k
        [U, S, ~] = svd(X, 'econ');
        singular = diag(S);
        U = U(:, 1:min(q, size(U,2)));
        singular = singular(1:size(U,2));
    else
        [U, S, ~] = svds(X, q);
        singular = diag(S);
        [singular, order] = sort(singular, 'descend');
        U = U(:, order);
    end

    if isempty(singular) || singular(1) == 0
        rank_tol = 0;
        r = 0;
    else
        rank_tol = max(size(X)) * eps(singular(1));
        r = sum(singular > rank_tol);
    end
    U_data = standardize_column_signs(U(:,1:r));

    H_cluster = zeros(n, k);
    H_cluster(:,1:r) = U_data;
    H = deterministic_completion(U_data, n, k);

    orth_error = norm(H' * H - eye(k), 'fro');
    if orth_error > 1e-10
        error('initialize_latent_view:OrthogonalityFailure', ...
              'Failed to construct an orthonormal initialization (error %.3g).', ...
              orth_error);
    end

    info = struct('data_rank', r, 'requested_rank', k, ...
                  'completed_columns', k-r, 'rank_tolerance', rank_tol, ...
                  'orthogonality_error', orth_error);
end

function Q = deterministic_completion(U, n, k)
    Q = U;
    tol = 100 * eps(max(1, sqrt(n)));

    % Deterministic DCT-II-like candidates avoid consuming the experiment's
    % random stream and do not require forming an N-by-N null-space matrix.
    sample = (0:n-1)' + 0.5;
    frequency = 0;
    max_frequency = max(4*k, k + 16);
    while size(Q,2) < k && frequency < max_frequency
        candidate = cos(pi * sample * frequency / n);
        candidate = orthogonalize_twice(candidate, Q);
        candidate_norm = norm(candidate);
        if candidate_norm > tol
            Q(:,end+1) = candidate / candidate_norm; %#ok<AGROW>
        end
        frequency = frequency + 1;
    end

    % Extremely pathological overlaps fall back to deterministic coordinate
    % vectors, still without changing MATLAB's RNG state.
    coordinate = 1;
    while size(Q,2) < k && coordinate <= n
        candidate = zeros(n,1);
        candidate(coordinate) = 1;
        candidate = orthogonalize_twice(candidate, Q);
        candidate_norm = norm(candidate);
        if candidate_norm > tol
            Q(:,end+1) = candidate / candidate_norm; %#ok<AGROW>
        end
        coordinate = coordinate + 1;
    end
    if size(Q,2) < k
        error('initialize_latent_view:CompletionFailure', ...
              'Could not complete %d columns in R^%d.', k, n);
    end
    Q = standardize_column_signs(Q(:,1:k));
end

function x = orthogonalize_twice(x, Q)
    if isempty(Q), return; end
    x = x - Q * (Q' * x);
    x = x - Q * (Q' * x);
end

function Q = standardize_column_signs(Q)
    for j = 1:size(Q,2)
        [~, pivot] = max(abs(Q(:,j)));
        if Q(pivot,j) < 0, Q(:,j) = -Q(:,j); end
    end
end

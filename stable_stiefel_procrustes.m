function [H, info] = stable_stiefel_procrustes(M, H_ref)
%STABLE_STIEFEL_PROCRUSTES Solve max tr(H'*M), subject to H'*H=I.
% For a rank-deficient M, select an optimizer closest to H_ref.  This is a
% tie-break inside the original optimum set, not a perturbation of M.

    if ~isnumeric(M) || ~isreal(M) || any(~isfinite(M(:)))
        error('stable_stiefel_procrustes:InvalidM', ...
              'M must be a finite real numeric matrix.');
    end
    [n,k] = size(M);
    if n < k
        error('stable_stiefel_procrustes:InvalidShape', ...
              'The Stiefel constraint requires size(M,1) >= size(M,2).');
    end
    if nargin < 2 || ~isequal(size(H_ref),[n,k]) || ...
            ~isnumeric(H_ref) || ~isreal(H_ref) || any(~isfinite(H_ref(:)))
        error('stable_stiefel_procrustes:InvalidReference', ...
              'H_ref must be a finite real matrix with the same size as M.');
    end
    ref_error = norm(H_ref'*H_ref-eye(k),'fro');
    if ref_error > 1e-8
        error('stable_stiefel_procrustes:InfeasibleReference', ...
              'H_ref must satisfy H_ref''*H_ref=I (error %.3g).',ref_error);
    end

    [U,S,V] = svd(M,'econ');
    singular = diag(S);
    if isempty(singular) || singular(1) == 0
        rank_tol = 0;
        r = 0;
    else
        rank_tol = max(size(M))*eps(singular(1));
        r = sum(singular > rank_tol);
    end

    Ur = U(:,1:r);
    Vr = V(:,1:r);
    if r == k
        H = Ur*Vr';
        reference_rank = 0;
    else
        V0 = V(:,r+1:k);
        p = k-r;

        % Among primary optima, maximize tr(H'*H_ref), equivalently minimize
        % ||H-H_ref||_F.  Only the null singular subspace is free.
        B = H_ref*V0;
        B = project_twice(B,Ur);
        [Ub,Sb,Vb] = svd(B,'econ');
        sb = diag(Sb);
        if isempty(sb) || sb(1) == 0
            b_tol = 0;
            reference_rank = 0;
        else
            b_tol = max(size(B))*eps(sb(1));
            reference_rank = sum(sb > b_tol);
        end

        Ubr = Ub(:,1:reference_rank);
        Vbr = Vb(:,1:reference_rank);
        Q = Ubr*Vbr';
        missing = p-reference_rank;
        if missing > 0
            E = deterministic_complement([Ur,Ubr],n,missing);
            W0 = Vb(:,reference_rank+1:p);
            Q = Q + E*W0';
        end
        H = Ur*Vr' + Q*V0';
    end

    orth_error = norm(H'*H-eye(k),'fro');
    if orth_error > 1e-9
        error('stable_stiefel_procrustes:OrthogonalityFailure', ...
              'Constructed factor is not orthonormal (error %.3g).',orth_error);
    end
    achieved = real(trace(H'*M));
    optimum = sum(singular);
    objective_gap = abs(achieved-optimum)/max(1,abs(optimum));
    if objective_gap > 1e-10
        error('stable_stiefel_procrustes:OptimalityFailure', ...
              'Procrustes objective gap is %.3g.',objective_gap);
    end

    info = struct('numerical_rank',r,'columns',k,'nullity',k-r, ...
                  'rank_tolerance',rank_tol, ...
                  'reference_projected_rank',reference_rank, ...
                  'orthogonality_error',orth_error, ...
                  'objective_value',achieved,'objective_optimum',optimum, ...
                  'relative_objective_gap',objective_gap, ...
                  'reference_distance',norm(H-H_ref,'fro'));
end

function X = project_twice(X,Q)
    if isempty(Q), return; end
    X = X-Q*(Q'*X);
    X = X-Q*(Q'*X);
end

function E = deterministic_complement(forbidden,n,count)
    E = zeros(n,0);
    sample = (0:n-1)'+0.5;
    frequency = 0;
    max_frequency = max(8*(count+size(forbidden,2)),32);
    tol = 100*eps(max(1,sqrt(n)));
    while size(E,2) < count && frequency < max_frequency
        x = cos(pi*sample*frequency/n);
        x = project_twice(x,[forbidden,E]);
        nx = norm(x);
        if nx > tol, E(:,end+1)=x/nx; end %#ok<AGROW>
        frequency = frequency+1;
    end
    coordinate = 1;
    while size(E,2) < count && coordinate <= n
        x = zeros(n,1); x(coordinate)=1;
        x = project_twice(x,[forbidden,E]);
        nx = norm(x);
        if nx > tol, E(:,end+1)=x/nx; end %#ok<AGROW>
        coordinate = coordinate+1;
    end
    if size(E,2) < count
        error('stable_stiefel_procrustes:CompletionFailure', ...
              'Could not construct the null-space completion.');
    end
end

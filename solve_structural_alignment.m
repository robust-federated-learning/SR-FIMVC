function [A, info] = solve_structural_alignment(Z, M, k)
% Solve min tr(A'*Z'*M*Z*A), subject to A'*Z'*Z*A=I.
% Whiten only positive eigenvalues; no ridge changes the constraint.
    % SVD avoids squaring the condition number through Z'*Z.
    [~,S,V]=svd(full(Z),'econ'); singular=diag(S);
    threshold=max(singular)*sqrt(eps)*max(1,sqrt(size(Z,2)));
    keep=singular>threshold;
    if nnz(keep)<k
        error('SRFIMVC:AnchorRankDeficient', ...
              'Numerically stable rank(Z)=%d is below k=%d; use more independent anchors.',nnz(keep),k);
    end
    R=bsxfun(@rdivide,V(:,keep),singular(keep)');
    B=full(R'*(Z'*apply_graph_operator(M,Z*R))); B=(B+B')/2;
    [E,D]=eig(B); [values,order]=sort(real(diag(D)),'ascend');
    A=R*E(:,order(1:k));
    % Correct accumulated rounding in the retained positive-rank subspace.
    for pass=1:2
        Q=Z*A; T=(Q'*Q); T=(T+T')/2;
        [Vg,Dg]=eig(T); dg=diag(Dg);
        if any(dg<=0), error('SRFIMVC:AlignmentInfeasible','Singular alignment Gram matrix.'); end
        A=A*(Vg*diag(1./sqrt(dg))*Vg');
    end
    info=struct('rank',nnz(keep),'orth_error',norm((Z*A)'*(Z*A)-eye(k),'fro'), ...
                'eigenvalues',values(1:k));
    if info.orth_error>1e-7
        error('SRFIMVC:AlignmentInfeasible','Cannot initialize the exact orthogonality constraint.');
    end
end

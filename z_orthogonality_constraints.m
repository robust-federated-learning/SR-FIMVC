function [c,ceq,GC,GCeq] = z_orthogonality_constraints(x,rows,cols,N,m,A)
% Analytic Jacobian for upper triangle of (ZA)'(ZA)-I.
    Z=sparse(rows,cols,x,N,m); Q=Z*A; k=size(A,2);
    [a,b]=find(triu(true(k))); indices=sub2ind([k,k],a,b);
    E=Q'*Q-eye(k); ceq=E(indices); c=[];
    if nargout>2
        GC=sparse(numel(x),0);
        GCeq=sparse(A(cols,a).*Q(rows,b)+Q(rows,a).*A(cols,b));
    end
end

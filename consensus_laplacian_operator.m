function L = consensus_laplacian_operator(Y)
% Eq. (18), represented by Y and degrees; never constructs Y*Y'.
% Also valid for nonnegative soft memberships, for numerical verification.
    validateattributes(Y,{'numeric','logical'},{'2d','real','finite','nonnegative','nonempty'});
    Y=sparse(double(Y));
    counts=full(sum(Y,1))';
    L=struct('kind','consensus_laplacian','labels',Y,'degree',full(Y*counts));
end

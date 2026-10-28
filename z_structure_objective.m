function [f, gradient, reconstruction, structure] = z_structure_objective(Z,X,U,w,M,A,lambda1)
% Eq. (9b) with W,U,P,A,G,L frozen. lambda2*||P||^2 is constant.
    residual=Z*U-X;
    weighted=bsxfun(@times,residual,w);
    Q=Z*A; MQ=apply_graph_operator(M,Q);
    reconstruction=sum(weighted(:).^2);
    structure=sum(sum(Q.*MQ));
    f=reconstruction+lambda1*structure;
    if nargout>1
        gradient=2*bsxfun(@times,residual,w.^2)*U'+2*lambda1*(MQ*A');
    end
end

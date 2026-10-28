function result=audit_H_gradient()
% Finite-difference audit of the H objective stated in the paper.
    rng(41); n=19; k=4; d=7; lambda_graph=.6; lambda_align=.3;
    X=randn(n,d); labels=randi(5,n,1);
    Y=sparse(1:n,labels,1,n,5); S=Y*Y'; L=diag(sum(S,2))-S;
    A=X*X'-lambda_graph*L;
    w=.2+2*rand(n,1); D=diag(w.^2);
    H=randn(n,k); B=randn(n,k);

    objective=@(Q) trace(Q'*A*D*Q)+lambda_align*trace(Q'*B);
    G_objective=(A*D+D*A)*H+lambda_align*B;
    G_paper=2*A*D*H+lambda_align*B;
    G_legacy=A*D*H+lambda_align*B;

    trials=30; e_obj=zeros(trials,1); e_paper=e_obj; e_legacy=e_obj;
    step=1e-6;
    for j=1:trials
        E=randn(n,k); E=E/norm(E,'fro');
        fd=(objective(H+step*E)-objective(H-step*E))/(2*step);
        scale=max(1,abs(fd));
        e_obj(j)=abs(fd-sum(sum(G_objective.*E)))/scale;
        e_paper(j)=abs(fd-sum(sum(G_paper.*E)))/scale;
        e_legacy(j)=abs(fd-sum(sum(G_legacy.*E)))/scale;
    end
    commutator=norm(A*D-D*A,'fro')/max(1,norm(A*D,'fro'));
    result=struct('objective_gradient_max_error',max(e_obj), ...
                  'paper_gradient_max_error',max(e_paper), ...
                  'legacy_gradient_max_error',max(e_legacy), ...
                  'relative_commutator',commutator);
    assert(result.objective_gradient_max_error<1e-6);
    assert(result.paper_gradient_max_error>1e-3);
    assert(result.legacy_gradient_max_error>1e-3);
    fprintf(['PASS: finite differences identify (AD+DA)H+align as the gradient.\n' ...
             'max relative directional error: objective %.3g, paper %.3g, legacy %.3g; commutator %.3g.\n'], ...
             result.objective_gradient_max_error,result.paper_gradient_max_error, ...
             result.legacy_gradient_max_error,result.relative_commutator);
end

function test_projected_Z_update()
    rng(84); N=48; m=8; k=3; d=5;
    cfg=struct('anchor_s',3,'anchor_sigma',1,'lambda1',1, ...
        'z_projected_max_iter',8,'z_projected_max_backtracks',16, ...
        'z_projected_tol',1e-8,'z_projected_step_scale',1);
    X=randn(N,d); [Z,U]=completion_anchor_weights(X,m,cfg);
    labels=one_hot(mod((1:N)',4)+1,4);
    L=consensus_laplacian_operator(labels);
    [A,~]=solve_structural_alignment(Z,L,k);
    w=0.5+rand(N,1);
    [Znew,Anew,info]=update_Z_projected(Z,X,U,w,L,A,cfg);
    assert(info.objective_after<=info.objective_before+1e-8);
    assert(max(abs(full(sum(Znew,2))-1))<1e-10);
    assert(min(nonzeros(Znew))>=0 && info.support_error==0);
    assert(norm((Znew*Anew)'*(Znew*Anew)-eye(k),'fro')<1e-7);
    assert(info.accepted && info.step_norm>0);

    cfg.lambda1=0;
    [Zplain,~,plain]=update_Z_projected(Z,X,U,w,L,A,cfg);
    assert(plain.objective_after<=plain.objective_before+1e-8);
    assert(norm(Znew-Zplain,'fro')>1e-8);
    assert(norm((Znew-Zplain)*U,'fro')>1e-8);
    fprintf('PASS: projected structural Z is feasible, descending and structure-sensitive.\n');
end

function test_server_consensus()
% Validate the server objective, constraints, legacy equivalence and
% invariance to paired client cluster permutations.
    here=fileparts(mfilename('fullpath')); addpath(here); rng(79);
    n=120; k=5; nv=3; clients=cell(1,nv); Hparts=cell(1,nv);
    for v=1:nv
        labels=randi(k,n,1); clients{v}.Y=one_hot(labels,k);
        [clients{v}.C,~]=qr(randn(k),0);
        Hparts{v}=clients{v}.Y*clients{v}.C;
    end
    Hc=cat(2,Hparts{:}); Y0=one_hot(randi(k,n,1),k);
    [Qt,~]=qr(randn(nv*k,k),0); C0=Qt';

    cfg=struct('server_consensus_mode','orthogonal_bcd','server_consensus_max_iter',50,'k',k);
    [Y,C,info]=update_server_consensus(Hc,Y0,C0,cfg);
    assert(info.objective_after>=info.objective_before-1e-10);
    assert(norm(C*C'-eye(k),'fro')<1e-9);
    assert(all(sum(Y,2)==1) && all(Y(:)==0 | Y(:)==1));

    cfg.server_consensus_mode='legacy';
    [Yl,Cl,~]=update_server_consensus(Hc,Y0,C0,cfg);
    Ctexp=stable_stiefel_procrustes(Hc'*Y0,C0'); Cexp=Ctexp';
    [~,idx]=max(Hc*Cexp',[],2); Yexp=one_hot(idx,k);
    assert(norm(Cl-Cexp,'fro')<1e-10 && isequal(Yl,Yexp));

    server=struct('Y',Y0,'C',C0);
    cfg.server_consensus_mode='orthogonal_bcd';
    s1=server_step_plus(server,clients,cfg);
    permuted=clients;
    for v=1:nv
        p=randperm(k);
        permuted{v}.Y=clients{v}.Y(:,p);
        permuted{v}.C=clients{v}.C(p,:);
        assert(norm(permuted{v}.Y*permuted{v}.C-Hparts{v},'fro')<1e-12);
    end
    s2=server_step_plus(server,permuted,cfg);
    assert(isequal(s1.Y,s2.Y) && norm(s1.C-s2.C,'fro')<1e-10);
    assert(norm(cat(2,s1.C_blocks{:})-s1.C,'fro')<1e-12);
    assert(s1.update_info.objective_after>=s1.update_info.objective_before-1e-10);
    fprintf('PASS: server consensus objective, orthogonality, legacy equivalence and permutation invariance.\n');
end

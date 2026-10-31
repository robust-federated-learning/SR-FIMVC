function test_mean_fill_local_approximation()
% Unit tests for paper-compatible initialization and Eq. (14) updates.
    here=fileparts(mfilename('fullpath')); addpath(here);
    X=[1,10,1000;3,999,2000;777,30,3000;5,50,4000];
    O=[1,1,0;1,0,0;0,1,0;1,1,0];
    [F,mu]=mean_fill_missing(X,O);
    assert(isequal(F(O==1),X(O==1)));
    assert(abs(mu(1)-3)<1e-12 && abs(mu(2)-30)<1e-12 && mu(3)==0);
    assert(abs(F(3,1)-3)<1e-12 && abs(F(2,2)-30)<1e-12);
    assert(all(F(:,3)==0));
    X2=X; X2(O==0)=-123456;
    F2=mean_fill_missing(X2,O);
    assert(isequal(F,F2));

    rng(67); n=80; k=6; [H,~]=qr(randn(n,k),0);
    labels=randi(k,n,1); Y0=full(sparse(1:n,labels,1,n,k));
    [C0,~]=qr(randn(k),0);
    cfg=struct('local_approx_mode','orthogonal_bcd','local_approx_max_iter',30);
    [Y,C,info]=update_local_approximation(H,Y0,C0,cfg);
    assert(info.objective_after<=info.objective_before+1e-10);
    assert(norm(C'*C-eye(k),'fro')<1e-9);
    assert(all(sum(Y,2)==1) && all(Y(:)==0 | Y(:)==1));

    cfg.local_approx_mode='legacy';
    [Yl,Cl,~]=update_local_approximation(H,Y0,C0,cfg);
    Cexp=stable_stiefel_procrustes(Y0'*H,C0);
    [~,idx]=max(H*Cexp',[],2); Yexp=full(sparse(1:n,idx,1,n,k));
    assert(norm(Cl-Cexp,'fro')<1e-10 && isequal(Yl,Yexp));

    Yempty=zeros(n,k); Yempty(:,1)=1;
    cfg.local_approx_mode='orthogonal_bcd';
    [Ye,Ce,ie]=update_local_approximation(H,Yempty,C0,cfg);
    assert(norm(Ce'*Ce-eye(k),'fro')<1e-9 && all(sum(Ye,2)==1));
    assert(ie.objective_after<=ie.objective_before+1e-10);
    fprintf('PASS: mask-aware mean fill and constrained local Y/C block updates.\n');
end

function test_H_gradient_update()
% Validate objective evaluation, gradient modes and monotone safeguard.
    here=fileparts(mfilename('fullpath')); addpath(here);
    audit_H_gradient(); rng(53);
    n=60; d=9; k=5; X=randn(n,d); w=.3+rand(n,1);
    labels=randi(6,n,1); Y=sparse(1:n,labels,1,n,6);
    L=diag(full(sum(Y*Y',2)))-Y*Y';
    H0=random_stiefel(n,k); target=randn(n,k); lg=.4; la=.2;
    A=X*X'-lg*L; D=diag(w.^2);
    explicit=trace(H0'*A*D*H0)+la*trace(H0'*target);
    implicit=h_representation_objective(H0,X,w,L,lg,target,la);
    assert(abs(explicit-implicit)<1e-9*max(1,abs(explicit)));

    cfg=struct('h_gradient_mode','objective','h_monotone_safeguard',true);
    [H,info]=update_H_representation(X,w,L,H0,target,lg,la,cfg);
    assert(norm(H'*H-eye(k),'fro')<1e-9);
    assert(info.objective_after>=info.objective_before-1e-10*max(1,abs(info.objective_before)));

    ADH=A*(D*H0); alignment=la*target;
    cfg.h_monotone_safeguard=false; cfg.h_gradient_mode='paper';
    [Hp,ip]=update_H_representation(X,w,L,H0,target,lg,la,cfg);
    Hpexp=stable_stiefel_procrustes(2*ADH+alignment,H0);
    assert(norm(Hp-Hpexp,'fro')<1e-10 && strcmp(ip.gradient_mode,'paper'));

    cfg.h_gradient_mode='legacy';
    [Hl,il]=update_H_representation(X,w,L,H0,target,lg,la,cfg);
    Hlexp=stable_stiefel_procrustes(ADH+alignment,H0);
    assert(norm(Hl-Hlexp,'fro')<1e-10 && strcmp(il.gradient_mode,'legacy'));
    fprintf('PASS: H objective, corrected update, compatibility modes and monotone safeguard.\n');
end

function Q=random_stiefel(n,k)
    [Q,~]=qr(randn(n,k),0);
end

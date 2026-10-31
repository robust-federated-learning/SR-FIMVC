function test_stable_H_update()
% Regression tests for the rank-aware Stiefel Procrustes update.
    here=fileparts(mfilename('fullpath')); addpath(here);
    rng(23);

    % Full rank: exactly the same polar solution as the original U*V'.
    M=randn(70,8); Href=random_stiefel(70,8);
    [H,info]=stable_stiefel_procrustes(M,Href);
    [U,~,V]=svd(M,'econ'); Hold=U*V';
    assert(norm(H-Hold,'fro') < 1e-10);
    assert(info.numerical_rank==8 && info.nullity==0);

    % Rank deficient: retain the exact nuclear-norm optimum and feasibility.
    A=random_stiefel(70,5); B=random_stiefel(8,5);
    M=A*diag([9,7,4,2,1])*B';
    [H,info]=stable_stiefel_procrustes(M,Href);
    assert(info.numerical_rank==5 && info.nullity==3);
    assert(norm(H'*H-eye(8),'fro') < 1e-9);
    assert(abs(trace(H'*M)-sum(svd(M))) < 1e-9);
    [H2,info2]=stable_stiefel_procrustes(M,Href);
    assert(norm(H-H2,'fro') < 1e-10);
    assert(info2.relative_objective_gap < 1e-10);

    % Zero proxy: every feasible H is primary-optimal; closest solution is Href.
    [Hz,infoz]=stable_stiefel_procrustes(zeros(70,8),Href);
    assert(norm(Hz-Href,'fro') < 1e-10);
    assert(infoz.numerical_rank==0 && infoz.reference_distance < 1e-10);

    % Pathological case where the projected reference itself is rank deficient.
    n=12; k=4; Href=eye(n,k); Ur=eye(n,2); Vr=[zeros(2,2);eye(2)];
    M=Ur*diag([3,1])*Vr';
    [Hp,infop]=stable_stiefel_procrustes(M,Href);
    assert(infop.numerical_rank==2 && infop.reference_projected_rank==0);
    assert(norm(Hp'*Hp-eye(k),'fro') < 1e-9);
    assert(abs(trace(Hp'*M)-4) < 1e-10);

    fprintf('PASS: stable H update; full-rank equivalence, rank-deficient optimum, zero and pathological cases.\n');
end

function Q=random_stiefel(n,k)
    [Q,~]=qr(randn(n,k),0);
end

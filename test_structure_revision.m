revisionRoot=fileparts(mfilename('fullpath'));
addpath(revisionRoot,'-begin');
diary(fullfile(revisionRoot,'unit-test-results.txt'));
rng(32);N=24;m=6;k=2;d=4;
config=struct('anchor_s',3,'anchor_sigma',1,'lambda1',1,'z_max_iter',80,'z_max_seconds',30,'z_constraint_tol',1e-7);
X=randn(N,d);[Z,U]=completion_anchor_weights(X,m,config);
labels=one_hot(mod((1:N)',3)+1,3);S=labels*labels';L=spdiags(sum(S,2),0,N,N)-S;
G=spdiags(double(mod((1:N)',5)~=0),0,N,N);M=G*L*G';
[A,alignment]=solve_structural_alignment(Z,M,k); assert(alignment.orth_error<1e-10);
w=0.5+rand(N,1);[f,grad]=z_structure_objective(Z,X,U,w,M,A,1);
[rows,cols,x]=find(Z);lin=sub2ind([N,m],rows,cols);
[~,eq,~,J]=z_orthogonality_constraints(x,rows,cols,N,m,A);
step=1e-6; ids=round(linspace(1,numel(x),8));
for j=ids
 xp=x;xm=x;xp(j)=xp(j)+step;xm(j)=xm(j)-step;
 Zp=sparse(rows,cols,xp,N,m);Zm=sparse(rows,cols,xm,N,m);
 fp=z_structure_objective(Zp,X,U,w,M,A,1);fm=z_structure_objective(Zm,X,U,w,M,A,1);
 assert(abs((fp-fm)/(2*step)-grad(lin(j)))<1e-5);
 [~,ep]=z_orthogonality_constraints(xp,rows,cols,N,m,A);[~,em]=z_orthogonality_constraints(xm,rows,cols,N,m,A);
 assert(norm((ep-em)/(2*step)-full(J(j,:))')<1e-7);
end
fprintf('PASS: objective and nonlinear-constraint analytic gradients match finite differences.\n');
[Znew,info]=update_Z_structured(Z,X,U,w,M,A,config);disp(info);
assert(info.accepted && info.objective_after<=info.objective_before+1e-8 && info.step_norm>1e-6);
assert(max(abs(sum(Znew,2)-1))<1e-7 && min(nonzeros(Znew))>=0 && info.orth_error<1e-6);
assert(nnz(Znew.*(spones(Z)==0))==0);
fprintf('PASS: constrained Z moves, objective decreases, support/simplex/orthogonality preserved.\n');
config.lambda1=0;[Zzero,infozero]=update_Z_structured(Z,X,U,w,M,A,config);
assert(infozero.accepted); delta=norm(Znew-Zzero,'fro');assert(delta>1e-5);
fprintf('PASS: lambda1 changes the optimized Z, delta=%g (same frozen A and constraints).\n',delta);
% Confirm this change propagates into feature reconstruction with fixed U.
assert(norm((Znew-Zzero)*U,'fro')>1e-6);
fprintf('PASS: structural Z update changes ZU reconstruction.\n');
% Verify actual client-level propagation into missing entries.
cfg=struct('k',k,'eps_w',1e-6,'mu_ridge',1e-3,'lambda_graph',0.5, ...
 'lambda_align',0.1,'lambda2',1,'anchor_m',m,'anchor_s',3,'R_anchor',4, ...
 'structure_z_enabled',true,'lambda1',1,'z_max_iter',80,'z_max_seconds',30,'z_constraint_tol',1e-7);
rng(51);O=double(rand(N,d)>0.25);O(1:3,:)=0;Xprev=O.*X;
[H,~]=qr(randn(N,k),0);Y=full(one_hot(mod((1:N)',k)+1,k));
rng(63);[~,~,~,Xstruct,~,state]=client_step_RWC(X,O,Xprev,L,H,Y,eye(k),eye(k),ones(N,1),cfg);
cfg.lambda1=0;rng(63);[~,~,~,Xplain]=client_step_RWC(X,O,Xprev,L,H,Y,eye(k),eye(k),ones(N,1),cfg);
assert(isequal(Xstruct(O==1),X(O==1)) && isequal(Xplain(O==1),X(O==1)));
assert(norm(Xstruct-Xplain,'fro')>1e-8);
fprintf('PASS: actual client completion changes with lambda1, delta=%g, observed entries unchanged.\n',norm(Xstruct-Xplain,'fro'));

diary off;

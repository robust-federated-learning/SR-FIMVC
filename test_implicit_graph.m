revisionRoot=fileparts(mfilename('fullpath'));
addpath(revisionRoot,'-begin');
diary(fullfile(revisionRoot,'implicit-unit-results.txt'));
rng(85);count=0;maxerr=0;
for N=[12,33]
 for k=[1,4,7]
  for mode=1:2
   if mode==1,Y=one_hot(randi(k,N,1),k);else,Y=rand(N,k);Y=Y./sum(Y,2);Y(1,:)=0;end
   S=Y*Y';L=spdiags(sum(S,2),0,N,N)-S;op=consensus_laplacian_operator(Y);
   for width=[1,5]
    B=randn(N,width);expected=L*B;actual=apply_graph_operator(op,B);
    err=norm(expected-actual,'fro')/max(1,norm(expected,'fro'));assert(err<1e-12);maxerr=max(maxerr,err);count=count+1;
    for mask={zeros(N,1),ones(N,1),double(rand(N,1)>.3)}
     G=spdiags(mask{1},0,N,N);M=G*L*G';mo=masked_graph_operator(op,mask{1});
     assert(norm(apply_graph_operator(mo,B)-M*B,'fro')/max(1,norm(M*B,'fro'))<1e-12);
    end
   end
  end
 end
end
fprintf('PASS: %d hard/soft/empty-cluster graph products, max relative error=%g; masked products also match.\n',count,maxerr);
% Fixed-input structure objective/gradient equivalence.
N=24;m=6;k=2;d=4;X=randn(N,d);cfg=struct('anchor_s',3,'anchor_sigma',1);
[Z,U]=completion_anchor_weights(X,m,cfg);Yg=one_hot(mod((1:N)',2)+1,2);
S=Yg*Yg';L=spdiags(sum(S,2),0,N,N)-S;op=consensus_laplacian_operator(Yg);
mask=double(mod((1:N)',5)~=0);G=spdiags(mask,0,N,N);M=G*L*G';mo=masked_graph_operator(op,mask);
A=solve_structural_alignment(Z,M,k);Ai=solve_structural_alignment(Z,mo,k);
assert(norm(A*A'-Ai*Ai','fro')<1e-7);
w=.5+rand(N,1);[f,g,r,s]=z_structure_objective(Z,X,U,w,M,A,1);[fi,gi,ri,si]=z_structure_objective(Z,X,U,w,mo,A,1);
assert(abs(f-fi)<1e-10 && norm(g-gi,'fro')<1e-9);
fprintf('PASS: structural alignment subspace, objective and gradient match explicit masked graph.\n');
% Actual client with the constrained Z pathway active. No wall-clock stop.
cfg=struct('k',k,'eps_w',1e-6,'mu_ridge',1e-3,'lambda_graph',.5,'lambda_align',.1,'lambda2',1,'lambda1',1,'anchor_m',m,'anchor_s',3,'R_anchor',4,'structure_z_enabled',true,'z_max_iter',3,'z_max_seconds',60,'z_constraint_tol',1e-7);
O=double(rand(N,d)>.2);O(1:3,:)=0;Xprev=X.*O;[H,~]=qr(randn(N,k),0);Y=full(one_hot(mod((1:N)',k)+1,k));
rng(52);[Ye,Ce,He,Xe,we,se]=client_step_RWC(X,O,Xprev,L,H,Y,eye(k),eye(k),ones(N,1),cfg);
rng(52);[Yi,Ci,Hi,Xi,wi,si]=client_step_RWC(X,O,Xprev,op,H,Y,eye(k),eye(k),ones(N,1),cfg);
assert(norm(Xe-Xi,'fro')<1e-6 && norm(He-Hi,'fro')<1e-6 && norm(Ce-Ci,'fro')<1e-6 && isequal(Ye,Yi));
fprintf('PASS: constrained client explicit/implicit: completion delta=%g, H delta=%g, labels identical.\n',norm(Xe-Xi,'fro'),norm(He-Hi,'fro'));
% Memory at a moderate N; larger implicit-only check never constructs pairs.
Nmem=3000;kmem=10;Ym=one_hot(mod((1:Nmem)',kmem)+1,kmem);
Sm=Ym*Ym';Lm=spdiags(sum(Sm,2),0,Nmem,Nmem)-Sm;Om=consensus_laplacian_operator(Ym);
memE=whos('Lm');memI=whos('Om');fprintf('MEMORY N=%d k=%d: explicit L=%d bytes, operator=%d bytes, ratio=%g.\n',Nmem,kmem,memE.bytes,memI.bytes,memE.bytes/memI.bytes);
Bmem=randn(Nmem,8);te=timeit(@() apply_graph_operator(Lm,Bmem));ti=timeit(@() apply_graph_operator(Om,Bmem));fprintf('PRODUCT TIMING explicit=%gs implicit=%gs.\n',te,ti);
Ylarge=one_hot(mod((1:30000)',31)+1,31);Olarge=consensus_laplacian_operator(Ylarge);out=apply_graph_operator(Olarge,ones(30000,2));assert(norm(out,'fro')<1e-7);big=whos('Olarge');fprintf('PASS: N=30000 implicit-only product; operator bytes=%d, no pairwise matrix constructed.\n',big.bytes);

diary off;

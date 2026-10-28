function [Znew, info] = update_Z_structured(Z,X,U,w,M,A,config)
% Reference inexact constrained Z step for Eq. (9b), fixed nearest-anchor support.
% Interior-point + sparse Jacobians + L-BFGS avoids a dense nvar-by-nvar Hessian.
% Requires MATLAB Optimization Toolbox. No unconstrained/greedy fallback.
    if exist('fmincon','file')~=2
        error('SRFIMVC:OptimizationToolboxRequired', ...
              'The structural Z revision requires Optimization Toolbox (fmincon).');
    end
    if isfield(config,'lambda1'), lambda1=config.lambda1; else, lambda1=1; end
    if isfield(config,'z_max_iter'), maxiter=config.z_max_iter; else, maxiter=20; end
    if isfield(config,'z_max_seconds'), maxseconds=config.z_max_seconds; else, maxseconds=10; end
    if isfield(config,'z_constraint_tol'), tol=config.z_constraint_tol; else, tol=1e-7; end
    validateattributes(lambda1,{'numeric'},{'real','finite','scalar','nonnegative'});
    validateattributes(maxiter,{'numeric'},{'real','finite','scalar','integer','positive'});
    validateattributes(maxseconds,{'numeric'},{'real','finite','scalar','positive'});
    validateattributes(tol,{'numeric'},{'real','finite','scalar','positive'});
    [N,m]=size(Z); [rows,cols,x0]=find(Z); nvar=numel(x0);
    lin=sub2ind([N,m],rows,cols);
    Aeq=sparse(rows,(1:nvar)',1,N,nvar); beq=ones(N,1);
    [f0,~,r0,s0]=z_structure_objective(Z,X,U,w,M,A,lambda1);
    [~,e0]=z_orthogonality_constraints(x0,rows,cols,N,m,A);
    if max(abs(e0))>tol || max(abs(sum(Z,2)-1))>tol || any(x0<0)
        error('SRFIMVC:ZStartInfeasible','Z/A must be feasible before the constrained Z update.');
    end
    scale=max(1,abs(f0));
    solve_timer=tic;
    options=optimoptions('fmincon','Algorithm','interior-point', ...
        'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true, ...
        'HessianApproximation','lbfgs','Display','off','OutputFcn',@time_limit, ...
        'MaxIterations',maxiter,'MaxFunctionEvaluations',max(200,20*maxiter), ...
        'ConstraintTolerance',tol,'OptimalityTolerance',1e-7,'StepTolerance',1e-12);
    [x,~,exitflag,output]=fmincon(@objective,x0,[],[],Aeq,beq, ...
        zeros(nvar,1),ones(nvar,1),@constraints,options);
    % Backtrack toward the feasible start if a limited-iteration candidate
    % has not yet met the nonlinear constraints. Linear support/simplex
    % constraints are preserved along the segment, up to solver tolerance.
    x=max(0,min(1,x)); raw=x; alpha=1; accepted=false;
    raw_violation=NaN;
    for backtrack=0:40
        x=x0+alpha*(raw-x0);
        candidate=sparse(rows,cols,x,N,m);
        [fc,~,rc,sc]=z_structure_objective(candidate,X,U,w,M,A,lambda1);
        [~,ec]=constraints(x);
        violation=max([max(abs(Aeq*x-beq));max(abs(ec))]);
        if backtrack==0, raw_violation=violation; end
        accepted=isfinite(fc) && violation<=tol && fc<=f0+1e-12*scale;
        if accepted, break; end
        alpha=alpha/2;
    end
    if accepted
        Znew=candidate; f1=fc; r1=rc; s1=sc;
    else
        Znew=Z; f1=f0; r1=r0; s1=s0;
    end
    info=struct('accepted',accepted,'exitflag',exitflag,'iterations',output.iterations, ...
        'objective_before',f0,'objective_after',f1,'reconstruction_before',r0, ...
        'reconstruction_after',r1,'structure_before',s0,'structure_after',s1, ...
        'candidate_violation',violation,'raw_candidate_violation',raw_violation, ...
        'backtrack_alpha',alpha,'step_norm',norm(Znew-Z,'fro'), ...
        'orth_error',norm((Znew*A)'*(Znew*A)-eye(size(A,2)),'fro'), ...
        'solve_seconds',toc(solve_timer),'nvar',nvar,'lambda1',lambda1,'message',output.message);
    if ~accepted
        warning('SRFIMVC:ZCandidateRejected', ...
                'Z step rejected (exitflag=%d, constraint error=%g); retained feasible Z.',exitflag,violation);
    end
    function stop=time_limit(~,~,~)
        % Soft time budget: checked at solver callbacks, not inside a factorization.
        stop=toc(solve_timer)>=maxseconds;
    end
    function [f,g]=objective(values)
        trial=sparse(rows,cols,values,N,m);
        [f,grad]=z_structure_objective(trial,X,U,w,M,A,lambda1);
        f=f/scale; g=grad(lin)/scale;
    end
    function [c,ceq,GC,GCeq]=constraints(values)
        [c,ceq,GC,GCeq]=z_orthogonality_constraints(values,rows,cols,N,m,A);
    end
end

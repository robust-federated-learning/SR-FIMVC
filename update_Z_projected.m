function [Znew,Anew,info]=update_Z_projected(Z,X,U,w,M,A,config)
% Scalable inexact alternating Z/A update on a fixed anchor support.
% Z is updated by projected gradient on each row simplex. The paper's
% generalized eigenproblem then refreshes A and restores (Z*A)'*(Z*A)=I.
    if isfield(config,'lambda1'), lambda1=config.lambda1; else, lambda1=1; end
    if isfield(config,'z_projected_max_iter'), max_iter=config.z_projected_max_iter; else, max_iter=5; end
    if isfield(config,'z_projected_max_backtracks'), max_backtracks=config.z_projected_max_backtracks; else, max_backtracks=12; end
    if isfield(config,'z_projected_tol'), tolerance=config.z_projected_tol; else, tolerance=1e-6; end
    if isfield(config,'z_projected_step_scale'), step_scale=config.z_projected_step_scale; else, step_scale=1; end
    validateattributes(lambda1,{'numeric'},{'real','finite','scalar','nonnegative'});
    validateattributes(max_iter,{'numeric'},{'real','finite','scalar','integer','positive'});
    validateattributes(max_backtracks,{'numeric'},{'real','finite','scalar','integer','positive'});
    validateattributes(tolerance,{'numeric'},{'real','finite','scalar','positive'});
    validateattributes(step_scale,{'numeric'},{'real','finite','scalar','positive'});

    timer=tic;
    [N,m]=size(Z); k=size(A,2);
    [rows,cols,x]=find(Z); nvar=numel(x);
    groups=accumarray(rows,(1:nvar)',[N,1],@(q){q},{[]});
    support=sparse(rows,cols,1,N,m);
    Zcur=sparse(rows,cols,x,N,m);
    Acur=A;
    [f0,~,r0,s0]=z_structure_objective(Zcur,X,U,w,M,Acur,lambda1);
    fcur=f0; rcur=r0; scur=s0;
    total_backtracks=0; accepted_steps=0; converged=false;
    last_step=0; last_projected_norm=0;

    for iteration=1:max_iter
        residual=Zcur*U-X;
        Q=Zcur*Acur;
        MQ=apply_graph_operator(M,Q);
        grad_reconstruction=2*(w(rows).^2).*sum(residual(rows,:).*U(cols,:),2);
        grad_structure=2*lambda1*sum(MQ(rows,:).*Acur(cols,:),2);
        gradient=grad_reconstruction+grad_structure;

        local_graph_scale=norm(MQ,'fro')/max(norm(Q,'fro'),eps);
        lipschitz=2*max(w.^2)*norm(U,2)^2+ ...
                  2*lambda1*local_graph_scale*norm(Acur,2)^2;
        step=step_scale/max(1,lipschitz);
        accepted=false;

        for backtrack=1:max_backtracks
            trial_values=x-step*gradient;
            for i=1:N
                ids=groups{i};
                if ~isempty(ids)
                    trial_values(ids)=project_simplex(trial_values(ids));
                end
            end
            projected_norm=norm(trial_values-x);
            if projected_norm<=tolerance*max(1,norm(x))
                converged=true;
                break;
            end

            Ztrial=sparse(rows,cols,trial_values,N,m);
            try
                [Atrial,alignment]=solve_structural_alignment(Ztrial,M,k);
                [ftrial,~,rtrial,strial]=z_structure_objective( ...
                    Ztrial,X,U,w,M,Atrial,lambda1);
                objective_tolerance=1e-12*max(1,abs(fcur));
                accepted=isfinite(ftrial) && ftrial<=fcur+objective_tolerance;
            catch
                accepted=false;
            end
            if accepted
                x=trial_values; Zcur=Ztrial; Acur=Atrial;
                fcur=ftrial; rcur=rtrial; scur=strial;
                last_step=step; last_projected_norm=projected_norm;
                accepted_steps=accepted_steps+1;
                break;
            end
            step=step/2;
            total_backtracks=total_backtracks+1;
        end
        if converged || ~accepted
            break;
        end
    end

    Znew=Zcur; Anew=Acur;
    row_error=max(abs(full(sum(Znew,2))-1));
    orth_error=norm((Znew*Anew)'*(Znew*Anew)-eye(k),'fro');
    support_error=nnz(Znew)-nnz(Znew.*support);
    info=struct('solver','projected','accepted',accepted_steps>0, ...
        'iterations',iteration,'accepted_steps',accepted_steps, ...
        'backtracks',total_backtracks,'converged',converged, ...
        'objective_before',f0,'objective_after',fcur, ...
        'reconstruction_before',r0,'reconstruction_after',rcur, ...
        'structure_before',s0,'structure_after',scur, ...
        'step_norm',norm(Znew-Z,'fro'),'last_projected_norm',last_projected_norm, ...
        'last_step_size',last_step,'row_sum_error',row_error, ...
        'orth_error',orth_error,'support_error',support_error, ...
        'solve_seconds',toc(timer),'nvar',nvar,'lambda1',lambda1);
end

function projected=project_simplex(values)
    values=values(:);
    sorted=sort(values,'descend');
    cumulative=cumsum(sorted)-1;
    indices=(1:numel(values))';
    rho=find(sorted-cumulative./indices>0,1,'last');
    if isempty(rho)
        projected=ones(size(values))/numel(values);
    else
        theta=cumulative(rho)/rho;
        projected=max(values-theta,0);
        projected=projected/max(sum(projected),eps);
    end
end

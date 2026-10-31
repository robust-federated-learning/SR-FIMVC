function [X_completed,P,info]=update_completion_laplacian( ...
    X_base,X_rec,w,Omega,L,lambda2,config)
% Direct graph-regularized completion on missing entries only.
    if isfield(config,'completion_graph_weight')
        lambda_c=config.completion_graph_weight;
    else
        lambda_c=0;
    end
    if isfield(config,'completion_graph_max_iter')
        max_iter=config.completion_graph_max_iter;
    else
        max_iter=10;
    end
    if isfield(config,'completion_graph_max_backtracks')
        max_backtracks=config.completion_graph_max_backtracks;
    else
        max_backtracks=12;
    end
    if isfield(config,'completion_graph_tol')
        tolerance=config.completion_graph_tol;
    else
        tolerance=1e-6;
    end
    if isfield(config,'completion_graph_normalize')
        normalize_graph=config.completion_graph_normalize;
    else
        normalize_graph=true;
    end
    validateattributes(lambda_c,{'numeric'},{'real','finite','scalar','nonnegative'});
    validateattributes(max_iter,{'numeric'},{'real','finite','scalar','integer','positive'});
    validateattributes(max_backtracks,{'numeric'},{'real','finite','scalar','integer','positive'});

    missing=double(Omega==0);
    shrink=(w.^2)./(w.^2+lambda2);
    P=missing.*bsxfun(@times,X_rec-X_base,shrink);
    X_completed=X_base+P;

    graph_scale=1;
    graph_bound=0;
    if isstruct(L) && isfield(L,'degree')
        max_degree=max(L.degree);
        graph_bound=2*max_degree;
        if normalize_graph
            graph_scale=max(1,max_degree);
        end
    elseif isnumeric(L)
        if issparse(L)
            graph_bound=normest(L);
        else
            graph_bound=norm(L,2);
        end
        if normalize_graph
            graph_scale=max(1,graph_bound);
        end
    end

    [objective_before,reconstruction_before,ridge_before,graph_before]= ...
        completion_objective(X_completed,P,X_rec,w,L,lambda2,lambda_c,graph_scale);
    objective_current=objective_before;
    accepted_steps=0; backtracks=0; converged=false; last_step=0;

    if lambda_c>0 && any(missing(:))
        missing_rows=any(missing,2);
        active_weight=max(w(missing_rows).^2);
        scaled_graph_bound=graph_bound/max(graph_scale,eps);
        lipschitz=2*(active_weight+lambda2+lambda_c*scaled_graph_bound);
        initial_step=0.9/max(lipschitz,eps);

        for iteration=1:max_iter
            graph_gradient=apply_graph_operator(L,X_completed)/graph_scale;
            gradient=2*bsxfun(@times,X_completed-X_rec,w.^2)+ ...
                     2*lambda2*P+2*lambda_c*graph_gradient;
            gradient=missing.*gradient;
            step=initial_step;
            accepted=false;
            for trial=1:max_backtracks
                P_trial=missing.*(P-step*gradient);
                X_trial=X_base+P_trial;
                objective_trial=completion_objective( ...
                    X_trial,P_trial,X_rec,w,L,lambda2,lambda_c,graph_scale);
                if isfinite(objective_trial) && ...
                        objective_trial<=objective_current+1e-12*max(1,abs(objective_current))
                    accepted=true;
                    break;
                end
                step=step/2;
                backtracks=backtracks+1;
            end
            if ~accepted
                break;
            end
            relative_step=norm(P_trial-P,'fro')/max(1,norm(P,'fro'));
            P=P_trial; X_completed=X_trial;
            objective_current=objective_trial;
            accepted_steps=accepted_steps+1;
            last_step=step;
            if relative_step<=tolerance
                converged=true;
                break;
            end
        end
    else
        iteration=0;
    end

    [objective_after,reconstruction_after,ridge_after,graph_after]= ...
        completion_objective(X_completed,P,X_rec,w,L,lambda2,lambda_c,graph_scale);
    observed_error=norm(Omega.*(X_completed-X_base),'fro');
    info=struct('enabled',lambda_c>0,'lambda',lambda_c, ...
        'iterations',iteration,'accepted_steps',accepted_steps, ...
        'backtracks',backtracks,'converged',converged,'last_step_size',last_step, ...
        'objective_before',objective_before,'objective_after',objective_after, ...
        'reconstruction_before',reconstruction_before, ...
        'reconstruction_after',reconstruction_after,'ridge_before',ridge_before, ...
        'ridge_after',ridge_after,'graph_before',graph_before, ...
        'graph_after',graph_after,'graph_scale',graph_scale, ...
        'observed_error',observed_error);
end

function [value,reconstruction,ridge,graph]=completion_objective( ...
    X_completed,P,X_rec,w,L,lambda2,lambda_c,graph_scale)
    residual=bsxfun(@times,X_completed-X_rec,w);
    reconstruction=sum(residual(:).^2);
    ridge=lambda2*sum(P(:).^2);
    if lambda_c>0
        LX=apply_graph_operator(L,X_completed)/graph_scale;
        graph=lambda_c*sum(sum(X_completed.*LX));
    else
        graph=0;
    end
    value=reconstruction+ridge+graph;
end

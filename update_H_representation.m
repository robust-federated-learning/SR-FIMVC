function [H,info]=update_H_representation(X,w,L,H_prev,align_target,lambda_graph,lambda_align,config)
%UPDATE_H_REPRESENTATION Rank-aware, matrix-free H update.
% mode='objective' uses the true gradient of Tr(H'*A*D*H)+align.
% mode='paper' reproduces Eq. (13); mode='legacy' reproduces the old code.
    if isfield(config,'h_gradient_mode'), mode=lower(char(config.h_gradient_mode));
    else, mode='objective'; end
    if isfield(config,'h_monotone_safeguard'), safeguard_raw=config.h_monotone_safeguard;
    else, safeguard_raw=true; end
    if ~isscalar(safeguard_raw) || ...
            ~(islogical(safeguard_raw) || (isnumeric(safeguard_raw) && isfinite(safeguard_raw)))
        error('update_H_representation:InvalidSafeguard', ...
              'h_monotone_safeguard must be a finite scalar logical/numeric value.');
    end
    safeguard=logical(safeguard_raw);
    if ~ismember(mode,{'objective','paper','legacy'})
        error('update_H_representation:InvalidMode', ...
              'h_gradient_mode must be objective, paper, or legacy.');
    end

    D_H=bsxfun(@times,H_prev,w.^2);
    AD_H=X*(X'*D_H)-lambda_graph*apply_graph_operator(L,D_H);
    alignment=lambda_align*align_target;
    switch mode
        case 'objective'
            A_H=X*(X'*H_prev)-lambda_graph*apply_graph_operator(L,H_prev);
            DA_H=bsxfun(@times,A_H,w.^2);
            proxy=AD_H+DA_H+alignment;
        case 'paper'
            proxy=2*AD_H+alignment;
        otherwise
            proxy=AD_H+alignment;
    end

    [candidate,polar_info]=stable_stiefel_procrustes(proxy,H_prev);
    before=h_representation_objective(H_prev,X,w,L,lambda_graph,align_target,lambda_align);
    candidate_value=h_representation_objective(candidate,X,w,L,lambda_graph,align_target,lambda_align);
    tolerance=1e-12*max(1,abs(before));
    H=candidate; after=candidate_value; alpha=1; accepted=true;

    if safeguard && candidate_value < before-tolerance
        accepted=false;
        for j=1:30
            alpha=2^(-j);
            raw=(1-alpha)*H_prev+alpha*candidate;
            [trial,~]=stable_stiefel_procrustes(raw,H_prev);
            trial_value=h_representation_objective(trial,X,w,L,lambda_graph,align_target,lambda_align);
            if trial_value >= before-tolerance
                H=trial; after=trial_value; accepted=true; break;
            end
        end
        if ~accepted
            H=H_prev; after=before; alpha=0;
        end
    end

    info=polar_info;
    info.gradient_mode=mode;
    info.safeguard_enabled=safeguard;
    info.candidate_objective=candidate_value;
    info.objective_before=before;
    info.objective_after=after;
    info.objective_change=after-before;
    info.accepted=accepted;
    info.backtrack_alpha=alpha;
end

function [Y_v, C_v, H_v, X_completed, K_diag_weights, anchor_state] = ...
    client_step_RWC(X_incomplete, Omega, X_completed_prev, ...
                    L_sum, H_v_prev, Y_v_prev, C_v_prev, ...
                    C_server_v, K_diag_weights_prev, ...
                    config, anchor_state)

% Optional per-client state; old callers can omit it and request five outputs.
if nargin < 11 || isempty(anchor_state)
    anchor_state = struct('iteration',0);
end
anchor_state.iteration = anchor_state.iteration + 1;
if isfield(config,'R_anchor'), anchor_period=config.R_anchor; else, anchor_period=1; end
validateattributes(anchor_period, {'numeric'}, {'real','finite','scalar','integer','positive'});


k = config.k;
N = size(X_incomplete, 1);
eps_w = config.eps_w; 
mu_ridge = config.mu_ridge;
lambda_graph = config.lambda_graph;
lambda_align = config.lambda_align;
% lambda2 regularizes the missing-entry correction in paper Eq. (10).
% Keep direct calls compatible with configurations created before this fix.
if isfield(config, 'lambda2')
    lambda2 = config.lambda2;
else
    lambda2 = 1.0;
end
validateattributes(lambda2, {'numeric'}, ...
                   {'real','finite','scalar','nonnegative'}, ...
                   mfilename, 'config.lambda2');

if isfield(config,'structure_z_enabled'), structure_z_enabled=config.structure_z_enabled; else, structure_z_enabled=true; end


has_missing = any(Omega(:) == 0);

if has_missing


    
    m = config.anchor_m;
    if isempty(m), m = min(N, max(k, max(2,min(round(N*0.1),50)))); end
    validateattributes(m, {'numeric'}, {'real','finite','scalar','integer','>=',max(k,2),'<=',N});
    refresh = ~isfield(anchor_state,'Z') || ...
              ~isequal(size(anchor_state.Z),[N,m]) || ...
              mod(anchor_state.iteration-1,anchor_period)==0;
    if refresh
        % Paper Eq. (3): nearest-anchor Gaussian weights, row-normalized.
        [Z_v, U_v_temp, anchor_meta] = completion_anchor_weights(X_completed_prev,m,config);

        valid_idx = sum(Omega, 2) > 0;
        M_v = masked_graph_operator(L_sum,valid_idx);
        if structure_z_enabled
            % Exact feasible alignment for the new constrained Z subproblem.
            A_v = solve_structural_alignment(Z_v,M_v,k);
        else
            B_v = Z_v' * apply_graph_operator(M_v,Z_v);
            S_mat = Z_v' * Z_v + 1e-6 * eye(m);
    

            M_left = full((B_v + B_v')/2 + 1e-4*eye(m));
            M_right = full(S_mat);

            if m <= k + 2


                [V_eig, D_eig] = eig(M_left, M_right);
                eigenvalues = diag(D_eig);
                [~, sort_idx] = sort(real(eigenvalues), 'ascend');
        
                take_num = min(m, k);
                A_v = V_eig(:, sort_idx(1:take_num));
        

                if take_num < k
                    A_v = [A_v, zeros(m, k - take_num)];
                end
            else

                try
                    [A_v, ~] = eigs(M_left, M_right, k, 'smallestreal');
                catch

                    [V_eig, D_eig] = eig(M_left, M_right);
                    [~, sort_idx] = sort(real(diag(D_eig)), 'ascend');
                    A_v = V_eig(:, sort_idx(1:k));
                end
            end

        end
        anchor_state.Z = Z_v;
        anchor_state.A = A_v;
        anchor_state.U = U_v_temp;
        anchor_state.meta = anchor_meta;
        anchor_state.last_refresh = anchor_state.iteration;
    else
        Z_v = anchor_state.Z;
        A_v = anchor_state.A;
    end
    % U is refined every round, including rounds with cached Z and A.
    U_v_temp = anchor_state.U;



    X_rec = Z_v * U_v_temp;
    residuals_sq = sum(((X_incomplete - X_rec) .* Omega).^2, 2);
    w = 1.0 ./ sqrt(residuals_sq + eps_w); 
    

    is_missing = sum(Omega, 2) == 0;
    if any(is_missing)
        w(is_missing) = median(w(~is_missing)); 
    end
    if isfield(config,'rwc_enabled') && ~config.rwc_enabled
        w=ones(N,1);
    end
    W_sq = sparse(1:N, 1:N, w.^2, N, N);

    if refresh && structure_z_enabled
        if isfield(config,'z_solver'), z_solver=lower(char(config.z_solver));
        else, z_solver='projected'; end
        switch z_solver
            case 'projected'
                [Z_v,A_v,z_info] = update_Z_projected( ...
                    Z_v,X_completed_prev,U_v_temp,w,M_v,A_v,config);
            case 'fmincon_reference'
                [Z_v,z_info] = update_Z_structured( ...
                    Z_v,X_completed_prev,U_v_temp,w,M_v,A_v,config);
                z_info.solver='fmincon_reference';
            otherwise
                error('client_step_RWC:InvalidZSolver', ...
                      'z_solver must be projected or fmincon_reference.');
        end
        anchor_state.Z = Z_v;
        anchor_state.A = A_v;
        anchor_state.z_update = z_info;
    end 


    U_v = (Z_v' * W_sq * Z_v + mu_ridge * eye(m)) \ (Z_v' * W_sq * X_completed_prev);
    anchor_state.U = U_v;


    X_rec_final = Z_v * U_v;
    % Use the existing zero-filled observed matrix as the fixed baseline.
    % Never subtract hidden values retained by the synthetic data generator.
    X_base = Omega .* X_incomplete;
    [X_completed,~,completion_graph_info] = update_completion_laplacian( ...
        X_base,X_rec_final,w,Omega,L_sum,lambda2,config);
    anchor_state.completion_graph_update = completion_graph_info;

else



    
    X_completed = X_incomplete;
    w = ones(N, 1);
    K_diag_weights = w;
end

K_diag_weights = w; 




align_target = full(Y_v_prev) * C_server_v;
[H_v,h_update_info] = update_H_representation( ...
    X_completed,w,L_sum,H_v_prev,align_target, ...
    lambda_graph,lambda_align,config);
anchor_state.h_update = h_update_info;


[Y_v,C_v,local_approx_info] = update_local_approximation( ...
    H_v,Y_v_prev,C_v_prev,config);
anchor_state.local_approx_update = local_approx_info;

end

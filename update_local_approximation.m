function [Y,C,info]=update_local_approximation(H,Y_prev,C_prev,config)
%UPDATE_LOCAL_APPROXIMATION Minimize ||H-Y*C||_F^2 with one-hot Y, C'*C=I.
% orthogonal_bcd alternates the two exact constrained block updates until
% labels stabilize. legacy reproduces the former single C-then-Y sweep.
    if isfield(config,'local_approx_mode'), mode=lower(char(config.local_approx_mode));
    else, mode='orthogonal_bcd'; end
    if isfield(config,'local_approx_max_iter'), max_iter=config.local_approx_max_iter;
    else, max_iter=50; end
    validateattributes(max_iter,{'numeric'},{'real','finite','scalar','integer','positive'});
    if ~ismember(mode,{'orthogonal_bcd','legacy'})
        error('update_local_approximation:InvalidMode', ...
              'local_approx_mode must be orthogonal_bcd or legacy.');
    end
    [n,k]=size(H);
    if ~isequal(size(Y_prev),[n,k]) || ~isequal(size(C_prev),[k,k])
        error('update_local_approximation:InvalidShape','Y_prev or C_prev has incompatible size.');
    end
    if norm(C_prev'*C_prev-eye(k),'fro')>1e-8
        error('update_local_approximation:InfeasibleC','C_prev must satisfy C_prev''*C_prev=I.');
    end

    Y=Y_prev; C=C_prev;
    before=norm(H-Y*C,'fro')^2;
    converged=false; iterations=0;
    limit=1; if strcmp(mode,'orthogonal_bcd'), limit=max_iter; end
    for j=1:limit
        iterations=j;
        [C,~]=stable_stiefel_procrustes(full(Y)'*H,C);
        [~,labels]=max(H*C',[],2);
        Y_new=full(sparse(1:n,labels,1,n,k));
        converged=isequal(Y_new,Y);
        Y=Y_new;
        if converged, break; end
    end
    after=norm(H-Y*C,'fro')^2;
    tolerance=1e-10*max(1,abs(before));
    if after>before+tolerance
        error('update_local_approximation:ObjectiveIncrease', ...
              'Approximation objective increased from %.16g to %.16g.',before,after);
    end
    info=struct('mode',mode,'iterations',iterations,'converged',converged, ...
                'objective_before',before,'objective_after',after, ...
                'objective_change',after-before, ...
                'orthogonality_error',norm(C'*C-eye(k),'fro'), ...
                'empty_clusters',sum(sum(Y,1)==0));
end

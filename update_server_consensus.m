function [Y,C,info]=update_server_consensus(Hc,Y_prev,C_prev,config)
%UPDATE_SERVER_CONSENSUS Maximize Tr(Hc'*Y*C), Y one-hot, C*C'=I.
% orthogonal_bcd alternates exact C and Y block updates until labels are
% stable. legacy reproduces the former single C-then-Y server sweep.
    if isfield(config,'server_consensus_mode'), mode=lower(char(config.server_consensus_mode));
    else, mode='orthogonal_bcd'; end
    if isfield(config,'server_consensus_max_iter'), max_iter=config.server_consensus_max_iter;
    else, max_iter=50; end
    validateattributes(max_iter,{'numeric'},{'real','finite','scalar','integer','positive'});
    if ~ismember(mode,{'orthogonal_bcd','legacy'})
        error('update_server_consensus:InvalidMode', ...
              'server_consensus_mode must be orthogonal_bcd or legacy.');
    end
    [n,vk]=size(Hc); k=size(Y_prev,2);
    if ~isequal(size(Y_prev),[n,k]) || ~isequal(size(C_prev),[k,vk])
        error('update_server_consensus:InvalidShape','Y_prev or C_prev has incompatible size.');
    end
    if norm(C_prev*C_prev'-eye(k),'fro')>1e-8
        error('update_server_consensus:InfeasibleC','C_prev must satisfy C_prev*C_prev''=I.');
    end

    Y=Y_prev; C=C_prev;
    before=sum(sum((Y*C).*Hc));
    converged=false; iterations=0;
    limit=1; if strcmp(mode,'orthogonal_bcd'), limit=max_iter; end
    for j=1:limit
        iterations=j;
        % Transpose converts the row-Stiefel constraint C*C'=I into the
        % column-Stiefel form accepted by stable_stiefel_procrustes.
        [Ct,~]=stable_stiefel_procrustes(Hc'*Y,C');
        C=Ct';
        [~,labels]=max(Hc*C',[],2);
        Y_new=full(sparse(1:n,labels,1,n,k));
        converged=isequal(Y_new,Y);
        Y=Y_new;
        if converged, break; end
    end
    after=sum(sum((Y*C).*Hc));
    tolerance=1e-10*max(1,abs(before));
    if after<before-tolerance
        error('update_server_consensus:ObjectiveDecrease', ...
              'Server objective decreased from %.16g to %.16g.',before,after);
    end
    info=struct('mode',mode,'iterations',iterations,'converged',converged, ...
                'objective_before',before,'objective_after',after, ...
                'objective_change',after-before, ...
                'orthogonality_error',norm(C*C'-eye(k),'fro'), ...
                'empty_clusters',sum(sum(Y,1)==0));
end

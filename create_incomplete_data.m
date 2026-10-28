function [X_incomplete_list, Omega_list] = create_incomplete_data(X_list, PER)
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------


    if PER <= 0 || PER >= 1
        X_incomplete_list = X_list;
        Omega_list = cell(size(X_list));
        for i = 1:numel(X_list)
            Omega_list{i} = ones(size(X_list{i}));
        end
        fprintf('[data] PER=%.2f -> no missing, use full data.\n', PER);
        return;
    end

    nV = numel(X_list);
    X_incomplete_list = X_list;   
    Omega_list        = cell(1, nV);
    N = size(X_list{1}, 1);
    
    fprintf('[data] Generating MNAR missing with PER=%.2f (N=%d, V=%d)\n', PER, N, nV);

    for v = 1:nV
        Xv = X_list{v};
        [Nv, Dv] = size(Xv);
        

        norms = sqrt(sum(Xv.^2, 2)); 
        


        target_func = @(lam) mean(1 - exp(-lam * norms)) - PER;
        

        try
            lam_opt = fzero(target_func, [0, 1000]); 
        catch

            lam_opt = fzero(target_func, 1); 
        end
        

        P_missing = 1 - exp(-lam_opt * norms);
        

        missing_mask = rand(N, 1) < P_missing;
        

        Omega = ones(N, Dv);
        Omega(missing_mask, :) = 0;

        Omega_list{v}        = Omega;
        X_incomplete_list{v} = Xv;  

        actual_per = 1 - mean(Omega(:));
        fprintf('  [view %d] lambda=%.4f, actual PER=%.4f\n', v, lam_opt, actual_per);
    end

    % ---------------------------------------------------------------------

    % ---------------------------------------------------------------------
    miss_all = true(N, 1);
    for v = 1:nV
        miss_all = miss_all & (Omega_list{v}(:,1) == 0);
    end
    
    missing_all_idx = find(miss_all);
    if ~isempty(missing_all_idx)
        for i = 1:length(missing_all_idx)
            idx = missing_all_idx(i);

            restore_v = randi(nV);
            Omega_list{restore_v}(idx, :) = 1;
        end
        fprintf('  [warn] %d samples were missing in ALL views. Restored 1 random view for them.\n', length(missing_all_idx));
    end
end
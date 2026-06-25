function [X_incomplete_list, Omega_list] = create_incomplete_data(X_list, PER)
% -------------------------------------------------------------------------
% 匹配论文 Section V.A.2: Missing-View Protocol (MNAR)
% 机制: 样本缺失概率与其特征的 L2 范数正相关
% 公式 (Eq. 20): P(missing | x_i) = 1 - exp(-\lambda * ||x_i||_2)
% -------------------------------------------------------------------------

    % 边界情况：不引入缺失
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
        
        % 1. 计算每个样本的 L2 范数 ||x_i||_2
        norms = sqrt(sum(Xv.^2, 2)); 
        
        % 2. 匹配论文: "tuned via binary search to meet target missing rates"
        % 定义目标函数：平均缺失概率 - 目标缺失率 PER = 0
        target_func = @(lam) mean(1 - exp(-lam * norms)) - PER;
        
        % 使用 MATLAB 内置的 fzero 进行二分查找求解 \lambda_j
        try
            lam_opt = fzero(target_func, [0, 1000]); 
        catch
            % 如果找不到精确解，给一个保底的启发式 lambda
            lam_opt = fzero(target_func, 1); 
        end
        
        % 3. 匹配论文 Eq. (20): 计算每个样本的真实缺失概率
        P_missing = 1 - exp(-lam_opt * norms);
        
        % 4. 根据概率进行伯努利采样生成掩码
        missing_mask = rand(N, 1) < P_missing;
        
        % 生成最终的 Omega 矩阵
        Omega = ones(N, Dv);
        Omega(missing_mask, :) = 0;

        Omega_list{v}        = Omega;
        X_incomplete_list{v} = Xv;  

        actual_per = 1 - mean(Omega(:));
        fprintf('  [view %d] lambda=%.4f, actual PER=%.4f\n', v, lam_opt, actual_per);
    end

    % ---------------------------------------------------------------------
    % 安全兜底机制：确保没有样本在所有视图中全部缺失 (防止聚类崩溃)
    % ---------------------------------------------------------------------
    miss_all = true(N, 1);
    for v = 1:nV
        miss_all = miss_all & (Omega_list{v}(:,1) == 0);
    end
    
    missing_all_idx = find(miss_all);
    if ~isempty(missing_all_idx)
        for i = 1:length(missing_all_idx)
            idx = missing_all_idx(i);
            % 为该样本随机恢复一个视图
            restore_v = randi(nV);
            Omega_list{restore_v}(idx, :) = 1;
        end
        fprintf('  [warn] %d samples were missing in ALL views. Restored 1 random view for them.\n', length(missing_all_idx));
    end
end
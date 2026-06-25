function server = server_step_plus(server, clients, config)
    V = numel(clients);
    k = config.k;
    N = size(clients{1}.Y, 1);
    
    % --- 步骤 1: 匈牙利算法对齐 (必需!) ---
    % 以第一个视图或 server.C_ref 为基准对齐其他所有视图
    aligned_Yv = cell(1, V);
    aligned_Cv = cell(1, V);
    for v = 1:V
        % 计算成本矩阵并对齐
        cost_mat = clients{v}.C' * server.C_ref; 
        perm = custom_hungarian(-cost_mat); % 你需要实现或调用匈牙利算法
        aligned_Yv{v} = clients{v}.Y(:, perm);
        aligned_Cv{v} = clients{v}.C(perm, :);
    end

    % --- 步骤 2: 重建联合特征空间 Hc (公式 3) ---
    Hc = [];
    for v = 1:V
        Hc = [Hc, aligned_Yv{v} * aligned_Cv{v}]; % 
    end

    % --- 步骤 3: 严格按照论文公式更新 (公式 14, 15) ---
    % (a) 更新全局映射 C (正交 Procrustes 步骤)
    % 计算 Y' * Hc 的 SVD
    [U, ~, V_svd] = svd(server.Y' * Hc, 'econ');
    server.C = U * V_svd'; % 强制满足 CC' = I 

    % (b) 更新全局标签 Y (行向指派)
    [~, y_new_idx] = max(Hc * server.C', [], 2); % [cite: 253]
    server.Y = one_hot(y_new_idx, k);

    % --- 步骤 4: 刷新分块并更新参考基准 ---
    server.C_blocks = cell(1, V);
    for v = 1:V
        server.C_blocks{v} = server.C(:, (v-1)*k + (1:k)); % [cite: 148, 259]
    end
    % 更新参考质心用于下一轮对齐
    server.C_ref = server.C_blocks{1}; 
end
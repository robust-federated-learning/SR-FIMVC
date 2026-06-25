function [Xlist, gt, k, Mlist] = load_data_bypass(X_raw, gt_raw, folds, ~)
% load_data_bypass 严格对齐版
% 
% 核心逻辑：
% 1. 强制校验样本数 (N) 的一致性，严禁自动裁剪或补零。
% 2. 移除内部归一化，确保预处理权交由 main_fed_SR_FIMVC 统一管理。
% 3. 自动识别样本维度，确保特征矩阵以 [样本数 x 特征维数] 形式输出。

    % --- 1. 基础格式标准化 ---
    if ~iscell(X_raw)
        X_raw = {X_raw};
    end
    gt = gt_raw(:); % 确保是列向量
    N = numel(gt);  % 以标签总数作为唯一的样本数基准
    V = numel(X_raw);
    
    Xlist = cell(1, V);
    fprintf('开始数据对齐校验 (基准样本数 N = %d)...\n', N);

    % --- 2. 视图对齐校验 ---
    for v = 1:V
        X = X_raw{v};
        [m, n] = size(X);
        
        % 自动检测转置：如果行不匹配但列匹配，则进行转置
        if m ~= N && n == N
            X = X';
            [m, n] = size(X);
            fprintf('  - 视图 %d: 已自动转置以匹配样本数。\n', v);
        end
        
        % 🌟 致命性检查：严禁自动裁剪或补零 (这是导致 0.29 的主因)
        if m ~= N
            error('致命错误：视图 %d 的样本数 (%d) 与标签数 (%d) 不匹配！\n请检查原始数据加载逻辑，严禁在对齐不齐的情况下运行实验。', v, m, N);
        end
        
        % 确保数据类型
        Xlist{v} = double(X);
    end

    % --- 3. 计算聚类数 k ---
    k = numel(unique(gt)); 

    % --- 4. 构造完整观测掩码 (Mlist) ---
    % 注意：SR-FIMVC 的置缺逻辑在 main 函数中通过 PER 参数实现
    Mlist = repmat({true(N, 1)}, 1, V);

    % --- 5. 状态反馈 ---
    fprintf('✅ 数据对齐成功！[N=%d, V=%d, k=%d]\n', N, V, k);
    
    % 打印每类样本分布，方便排查是否存在标签污染
    tab = tabulate(gt);
    if any(tab(:, 2) ~= tab(1, 2))
        warning('数据类别不均衡：每类样本数分别为 [%s]', num2str(tab(:,2)'));
    else
        fprintf('  - 类别分布：完美均衡 (每类 %d 个样本)。\n', tab(1,2));
    end
end
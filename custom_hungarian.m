function assignment = custom_hungarian(costMatrix)
% =========================================================================
% 匈牙利算法实现 (基于 MATLAB matchpairs)
% 用于解决 SR-FIMVC 论文中公式 (13) 的标签排列对齐问题
% 
% 物理意义：
%   在服务端聚合时，通过最小化成本矩阵（或最大化相似度）来对齐不同客户端的聚类索引。
%
% 输入: 
%   costMatrix - 成本矩阵 (k x k)，通常传入的是 -Mv (负的质心相似度)
% 输出: 
%   assignment - 最优排列索引向量，用于 clients{v}.Y(:, assignment)
% =========================================================================

    % 1. 预处理：确保输入没有 NaN 或 Inf，防止计算崩溃 [cite: 166]
    costMatrix(isnan(costMatrix)) = 1e6;
    
    % 2. 使用 MATLAB 内置的线性指派函数 matchpairs
    % 该函数通过寻找二分图的最小权重匹配来实现匈牙利算法逻辑
    try
        % 1e6 作为未匹配的代价阈值
        pairs = matchpairs(costMatrix, 1e6);
        
        % 3. 将配对结果 (n x 2 矩阵) 转化为排列向量 (1 x k)
        num_clusters = size(costMatrix, 1);
        assignment = zeros(1, num_clusters);
        
        if ~isempty(pairs)
            % pairs(:,1) 是行索引（参考视图），pairs(:,2) 是列索引（当前视图）
            assignment(pairs(:,1)) = pairs(:,2);
        end
        
        % 安全补全：处理可能出现的未匹配索引
        unassigned_rows = find(assignment == 0);
        unassigned_cols = setdiff(1:num_clusters, assignment);
        if ~isempty(unassigned_rows)
            assignment(unassigned_rows) = unassigned_cols(1:length(unassigned_rows));
        end
        
    catch
        % 鲁棒性兜底：如果 matchpairs 出错，使用简单的贪心对齐
        [~, assignment] = max(-costMatrix, [], 2);
        % 确保 assignment 是唯一的排列
        if numel(unique(assignment)) < size(costMatrix, 1)
             assignment = 1:size(costMatrix, 1); % 最终降级：保持原序
        end
    end
end
function Y = one_hot(y, k)
% SAFE ONE-HOT (使用用户提供的安全版本)
% - y: 长度为 n 的标签向量
% - k: 类别数
% 返回:
% - Y: n x k 的 0/1 矩阵（稀疏表示）

    % ------- 输入规范化 -------
    if nargin < 2
        error('one_hot: 需要 y 和 k 两个输入');
    end
    y = y(:);            % 列向量
    n = numel(y);

    if ~isscalar(k) || ~isfinite(k) || k < 1
        error('one_hot: k 必须是 >=1 的有限标量');
    end
    k = round(double(k));

    % 把非有限/NaN 置为 1；把非整数四舍五入
    y(~isfinite(y)) = 1;
    y = round(double(y));

    % 夹紧到 [1, k] 范围
    y(y < 1) = 1;
    y(y > k) = k;

    % ------- 构造 one-hot -------
    rows = (1:n)';
    cols = y;
    vals = ones(n,1);
    Y = sparse(rows, cols, vals, n, k);

    % 如果需要 full 矩阵，请在此处取消注释：
    % Y = full(Y);
end
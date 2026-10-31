function Y = one_hot(y, k)







    if nargin < 2
        error('one_hot requires inputs y and k.');
    end
    y = y(:);
    n = numel(y);

    if ~isscalar(k) || ~isfinite(k) || k < 1
        error('one_hot: k must be a finite scalar greater than or equal to 1.');
    end
    k = round(double(k));


    y(~isfinite(y)) = 1;
    y = round(double(y));


    y(y < 1) = 1;
    y(y > k) = k;


    rows = (1:n)';
    cols = y;
    vals = ones(n,1);
    Y = sparse(rows, cols, vals, n, k);


    % Y = full(Y);
end

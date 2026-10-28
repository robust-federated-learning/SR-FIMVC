function [Xlist, gt, k, Mlist] = load_data_bypass(X_raw, gt_raw, folds, ~)

% 






    if ~iscell(X_raw)
        X_raw = {X_raw};
    end
    gt = gt_raw(:);
    N = numel(gt);
    V = numel(X_raw);
    
    Xlist = cell(1, V);


    for v = 1:V
        X = X_raw{v};
        [m, n] = size(X);
        

        if m ~= N && n == N
            X = X';
            [m, n] = size(X);
        end
        

        if m ~= N
            error('View %d has %d samples but %d labels. Check the data-loading logic.', v, m, N);
        end
        

        Xlist{v} = double(X);
    end


    k = numel(unique(gt)); 



    Mlist = repmat({true(N, 1)}, 1, V);


    

    tab = tabulate(gt);
    if any(tab(:, 2) ~= tab(1, 2))
        warning('Class counts are imbalanced: [%s]', num2str(tab(:,2)'));
    else
    end
end

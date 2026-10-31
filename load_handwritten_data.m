function [Xlist, gt, k] = load_handwritten_data(mat_path)
% Load and validate the Handwritten multi-view dataset.
    if nargin < 1 || isempty(mat_path)
        root = fileparts(mfilename('fullpath'));
        mat_path = fullfile(root, 'data', 'Handwritten.mat');
    end
    if exist(mat_path, 'file') ~= 2
        error('SRFIMVC:DataNotFound', ...
              'Handwritten data file was not found: %s', mat_path);
    end

    data = load(mat_path);
    if isfield(data, 'X')
        Xlist = data.X;
    elseif isfield(data, 'Xlist')
        Xlist = data.Xlist;
    else
        error('SRFIMVC:InvalidDataset', ...
              'Handwritten.mat must contain X or Xlist.');
    end

    if isfield(data, 'y')
        gt = data.y;
    elseif isfield(data, 'gt')
        gt = data.gt;
    elseif isfield(data, 'label')
        gt = data.label;
    else
        error('SRFIMVC:InvalidDataset', ...
              'Handwritten.mat must contain y, gt, or label.');
    end

    if ~iscell(Xlist) || isempty(Xlist)
        error('SRFIMVC:InvalidDataset', ...
              'Handwritten features must be stored as a nonempty cell array.');
    end
    gt = double(gt(:));
    sample_count = numel(gt);
    for view = 1:numel(Xlist)
        X = double(Xlist{view});
        if size(X,1) ~= sample_count && size(X,2) == sample_count
            X = X';
        end
        if size(X,1) ~= sample_count
            error('SRFIMVC:InvalidDataset', ...
                  'View %d has %d samples, but labels have %d.', ...
                  view, size(X,1), sample_count);
        end
        if any(~isfinite(X(:)))
            error('SRFIMVC:InvalidDataset', ...
                  'View %d contains NaN or Inf values.', view);
        end
        Xlist{view} = X;
    end
    k = numel(unique(gt));
end

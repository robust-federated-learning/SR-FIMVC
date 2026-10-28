function [Xlist, gt, k, Mlist] = load_dataset_by_name(name, folds, dataDir)



if nargin < 3 || isempty(dataDir), dataDir = 'data'; end
name = lower(string(name));

% normalize folds
if nargin < 2 || isempty(folds), folds = struct(); end
if ~isstruct(folds), folds = struct('PER', 0); end

switch name
    case {"mfeat","mfeat.mat"}
        error('The Mfeat dataset environment is not ready.');


    case {"mnist-usps","mnist_usps","mnistusps"}
        matPath = fullfile(dataDir, 'MNIST-USPS.mat');
        if ~exist(matPath, 'file'), error('Cannot find MNIST-USPS.mat'); end
        S = load(matPath);
        X1_3d = double(S.X1); X2_3d = double(S.X2);
        [N1, H1, W1] = size(X1_3d); 
        X1 = reshape(X1_3d, [N1, H1 * W1]);
        X2 = reshape(X2_3d, [size(X2_3d,1), size(X2_3d,2)*size(X2_3d,3)]);
        Xlist = {X1, X2};
        gt = double(S.Y(:));
        k = numel(unique(gt));
        Mlist = []; 


    case {"orl","orl.mat", "bbc","bbcsport", "caltech101-20","handwritten", ...
          "scene-15","scene15","outdoorscene", "100leaves", "landuse-21","landuse21"}
        
        nameToLoad = upper(char(name));

        if strcmpi(nameToLoad, 'BBC'), nameToLoad = 'BBCSPORT'; end
        if strcmpi(nameToLoad, 'OUTDOORSCENE'), nameToLoad = 'OutdoorScene'; end
        if strcmpi(nameToLoad, '100LEAVES'), nameToLoad = '100Leaves'; end
        if contains(nameToLoad, 'SCENE'), nameToLoad = 'Scene-15'; end
        if contains(nameToLoad, 'LANDUSE'), nameToLoad = 'LandUse-21'; end
        
        [ok, matPath] = resolve_mat(dataDir, char(name), [nameToLoad, '.mat']);
        if ~ok, error('Dataset %s.mat was not found.', nameToLoad); end
        

        S = load(matPath);
        

        Xlist = []; gt = [];


        if isfield(S, 'Xlist'),     Xlist = S.Xlist;
        elseif isfield(S, 'X'),     Xlist = S.X;
        elseif isfield(S, 'data'),  Xlist = S.data;
        elseif isfield(S, 'fea'),   Xlist = S.fea;
        end


        if isfield(S, 'gt_final'),  gt = S.gt_final; 
        elseif isfield(S, 'gt'),    gt = S.gt;
        elseif isfield(S, 'Y'),     gt = S.Y;
        elseif isfield(S, 'label'), gt = S.label;
        elseif isfield(S, 'gnd'),   gt = S.gnd;
        elseif isfield(S, 'y'),     gt = S.y;
        end


        if isfield(S, 'lenSmp') && ~iscell(Xlist) && ~isempty(Xlist)
            lenSmp = S.lenSmp;
            V_count = length(lenSmp);
            if size(Xlist, 2) == sum(lenSmp)
                X_tmp_list = cell(1, V_count);
                curr = 1;
                for v = 1:V_count
                    X_tmp_list{v} = Xlist(:, curr : curr + lenSmp(v) - 1);
                    curr = curr + lenSmp(v);
                end
                Xlist = X_tmp_list;
            end
        end


        if ~iscell(Xlist) && ~isempty(Xlist), Xlist = {Xlist}; end
        
        if isempty(Xlist) || isempty(gt)
            error('Expected features or labels were not found in %s.mat.', nameToLoad);
        end
        
        gt = gt(:); 
        k = numel(unique(gt));

    otherwise
        error('Unknown dataset "%s".', name);
end


V = numel(Xlist);
N_ref = numel(gt);

for v = 1:V
    X = double(Xlist{v});
    [m, n] = size(X);
    

    if n == N_ref && m ~= N_ref
        X = X'; 
        [m, n] = size(X); 
    end
    

    if m ~= N_ref
        error('Dataset %s view %d has %d samples but %d labels. Check the source MAT file.', name, v, m, N_ref);
    end
    
    Xlist{v} = X;
end

Mlist = repmat({true(N_ref,1)}, 1, V);
end

% ============= helper: resolve_mat =============
function [ok, matPath] = resolve_mat(dataDir, datasetName, fileName)
cands = {fullfile(dataDir, fileName), fullfile(dataDir, datasetName, fileName), ...
         fullfile(dataDir, lower(datasetName), fileName)};
ok = false; matPath = '';
for i = 1:numel(cands)
    if exist(cands{i}, 'file'), ok = true; matPath = cands{i}; return; end
end
end

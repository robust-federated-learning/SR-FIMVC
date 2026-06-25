function [Xlist, gt, k, Mlist] = load_dataset_by_name(name, folds, dataDir)
% 修正版：将所有 ! 替换为 MATLAB 标准的 ~ 运算符
% 修复：统一引入严格的样本对齐校验，彻底铲除静默裁剪带来的精度雪崩隐患。

if nargin < 3 || isempty(dataDir), dataDir = 'data'; end
name = lower(string(name));

% normalize folds
if nargin < 2 || isempty(folds), folds = struct(); end
if ~isstruct(folds), folds = struct('PER', 0); end

switch name
    case {"mfeat","mfeat.mat"}
        error('Mfeat 数据集加载环境未准备好。');

    % ===================== MNIST-USPS 专用加载分支 =====================
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

    % ===================== 通用加载列表 =====================
    case {"orl","orl.mat", "bbc","bbcsport", "caltech101-20","handwritten", ...
          "scene-15","scene15","outdoorscene", "100leaves", "landuse-21","landuse21"}
        
        nameToLoad = upper(char(name));
        % 名称映射
        if strcmpi(nameToLoad, 'BBC'), nameToLoad = 'BBCSPORT'; end
        if strcmpi(nameToLoad, 'OUTDOORSCENE'), nameToLoad = 'OutdoorScene'; end
        if strcmpi(nameToLoad, '100LEAVES'), nameToLoad = '100Leaves'; end
        if contains(nameToLoad, 'SCENE'), nameToLoad = 'Scene-15'; end
        if contains(nameToLoad, 'LANDUSE'), nameToLoad = 'LandUse-21'; end
        
        [ok, matPath] = resolve_mat(dataDir, char(name), [nameToLoad, '.mat']);
        if ~ok, error('未找到数据集 %s.mat。', nameToLoad); end
        
        % 加载到结构体 S
        S = load(matPath);
        
        % --- 🌟 解耦提取逻辑 🌟 ---
        Xlist = []; gt = [];

        % 1. 独立提取特征
        if isfield(S, 'Xlist'),     Xlist = S.Xlist;
        elseif isfield(S, 'X'),     Xlist = S.X;
        elseif isfield(S, 'data'),  Xlist = S.data;
        elseif isfield(S, 'fea'),   Xlist = S.fea;
        end

        % 2. 独立提取标签 (关键：包含 gt_final)
        if isfield(S, 'gt_final'),  gt = S.gt_final; 
        elseif isfield(S, 'gt'),    gt = S.gt;
        elseif isfield(S, 'Y'),     gt = S.Y;
        elseif isfield(S, 'label'), gt = S.label;
        elseif isfield(S, 'gnd'),   gt = S.gnd;
        elseif isfield(S, 'y'),     gt = S.y;
        end

        % === 特殊处理：Caltech101-20 的拼接矩阵拆分 ===
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

        % 确保 Xlist 是 cell 格式
        if ~iscell(Xlist) && ~isempty(Xlist), Xlist = {Xlist}; end
        
        if isempty(Xlist) || isempty(gt)
            error('提取失败：在 %s.mat 中未找到预期的特征或标签。', nameToLoad);
        end
        
        gt = gt(:); 
        k = numel(unique(gt));

    otherwise
        error('Unknown dataset "%s".', name);
end

% ---------------------- 5. 通用数据对齐逻辑 (严苛版) ----------------------
V = numel(Xlist);
N_ref = numel(gt);

for v = 1:V
    X = double(Xlist{v});
    [m, n] = size(X);
    
    % 自动转置 (d x N -> N x d)
    if n == N_ref && m ~= N_ref
        X = X'; 
        [m, n] = size(X); 
    end
    
    % 🚨 致命性检查：严禁自动裁剪或补零
    if m ~= N_ref
        error('致命错误：数据集 %s 视图 %d 的样本数 (%d) 与标签数 (%d) 不匹配！\n请检查原始 .mat 数据文件，严禁在错位的情况下运行实验。', name, v, m, N_ref);
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
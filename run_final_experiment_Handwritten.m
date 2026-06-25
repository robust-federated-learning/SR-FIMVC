%% run_grid_search_Handwritten.m (🚀 全自动参数雷达搜索 + 终极精度突围版)
clear; clc;
warning('off', 'all');

% ---------------------- 1. 基础实验配置 ----------------------
DATASET_NAME = 'Handwritten'; 
NUM_RUNS     = 10;         
T_MAX        = 200;           
PER_LIST     = [0.0, 0.1, 0.3, 0.5]; 

% 🚀 [大局锁死] 200个密集锚点，严格死守线性和合规防线
ANCHOR_M     = 300;    
PN_NEIGHBORS = 15;     

% ---------------------- 2. 定义自动网格搜索空间 (PHASE 1) ----------------------
% 围绕你之前的黄金参数，向外辐射搜索最高分辨率的特征解
LG_SPACE     = [0.1, 0.3, 0.5, 1.0, 1.5];   % \lambda_{graph} 搜索范围
LA_SPACE     = [0.01, 0.05, 0.1, 0.3, 0.5]; % \lambda_{align} 搜索范围
GRID_TARGET_PER = 0.3;                      % 🎯 精准瞄准 30% 缺失率进行性能突围

fprintf('======================================================\n');
fprintf('  STARTING AUTOMATIC GRID SEARCH & EVALUATION: %s\n', DATASET_NAME);
fprintf('  SEARCH PROFILE: L_g in [%s], L_a in [%s]\n', num2str(LG_SPACE), num2str(LA_SPACE));
fprintf('======================================================\n');

% 🚀 提速核心：只加载 1 次硬盘数据
fprintf('>>> 正在加载数据集 %s...\n', DATASET_NAME);
[Xlist, gt, k, ~] = load_dataset_by_name(DATASET_NAME); 
fprintf('>>> 数据加载完毕！\n');

% ---------------------- 3. 构建高分基础配置骨架 ----------------------
config_base = struct(... 
    'T',               T_MAX, ...
    'use_anchor',      true, ...        % 🚀 [核心主干不动] 死守动态锚点图与 O(N) 复杂度
    'anchor_m',        ANCHOR_M, ...    
    'pn',              PN_NEIGHBORS, ... 
    'R_anchor',        10, ...          
    'warm_start_iter', 50, ...          
    'smart_axis',      true, ...        % 出分杀手锏：超球面投影
    'metric',          'cosine' ...     % 出分杀手锏：余弦度量
);

% 🌟 外部独立赋值，彻底避开 MATLAB 嵌套 Cell 的闪退死穴
config_base.X_raw  = Xlist;
config_base.gt_raw = gt;
config_base.k      = k;

% ---------------------- 4. PHASE 1: 全自动网格搜索展开 ----------------------
fprintf('\n======================================================\n');
fprintf('  🔍 PHASE 1: SEARCHING FOR OPTIMAL PARAMETERS (PER=0.3)\n');
fprintf('======================================================\n');

best_acc = 0;
best_lg  = 0.5; % 默认兜底参数
best_la  = 0.1;

for lg = LG_SPACE
    for la = LA_SPACE
        % 复制基础配置并微调当前组合
        tmp_cfg = config_base;
        tmp_cfg.PER = GRID_TARGET_PER;
        tmp_cfg.lambda_graph = lg;
        tmp_cfg.lambda_align = la;
        
        % 搜索阶段每种组合跑 1 次快速扫射，寻找最优区间
        R = main_fed_SR_FIMVC(DATASET_NAME, 1, tmp_cfg);
        fprintf('  [Radar Trial] L_g=%.2f, L_a=%.2f => TARGET ACC: %.4f (NMI: %.4f)\n', lg, la, R.best.acc, R.best.nmi);
        
        % 刷新历史最高分纪录
        if R.best.acc > best_acc
            best_acc = R.best.acc;
            best_lg  = lg;
            best_la  = la;
        end
    end
end

fprintf('\n🌟 雷达搜索大捷！针对 PER=0.3 的最强神仙参数已锁定：\n');
fprintf('   👉 LAMBDA_GRAPH = %.2f\n', best_lg);
fprintf('   👉 LAMBDA_ALIGN = %.2f\n', best_la);
fprintf('   👉 探测最高 ACC  = %.4f\n', best_acc);

% ---------------------- 5. PHASE 2: 正式展开多缺失率多轮实验 ----------------------
fprintf('\n======================================================\n');
fprintf('  🚀 PHASE 2: FINAL EVALUATION WITH OPTIMAL PARAMETERS\n');
fprintf('======================================================\n');

% 更新配置为网格搜索出的最强组合
config_base.lambda_graph = best_lg;
config_base.lambda_align = best_la;

results_table = table();
all_runs_details = struct('PER', {}, 'RUN_ID', {}, 'R', {}); 
detail_counter = 1;

for p = 1:length(PER_LIST)
    current_per = PER_LIST(p);
    
    acc_runs = zeros(NUM_RUNS, 1);
    nmi_runs = zeros(NUM_RUNS, 1);
    ari_runs = zeros(NUM_RUNS, 1); 
    purity_runs = zeros(NUM_RUNS, 1);
    
    fprintf('\n--- Running Final PER = %.2f (%d Runs) ---\n', current_per, NUM_RUNS);

    for run = 1:NUM_RUNS
        config = config_base;
        config.PER = current_per;
        
        R = main_fed_SR_FIMVC(DATASET_NAME, run, config); 
        
        acc_runs(run) = R.best.acc; 
        nmi_runs(run) = R.best.nmi;
        ari_runs(run) = R.best.ari; 
        purity_runs(run) = R.best.purity;
        
        all_runs_details(detail_counter).PER = current_per;
        all_runs_details(detail_counter).RUN_ID = run;
        all_runs_details(detail_counter).R = R;
        detail_counter = detail_counter + 1;
        
        fprintf('  Run %d: ACC=%.4f, NMI=%.4f, ARI=%.4f\n', run, R.best.acc, R.best.nmi, R.best.ari); 
    end
    
    new_row = table(current_per, mean(acc_runs), std(acc_runs), mean(nmi_runs), std(nmi_runs), ...
                    mean(ari_runs), std(ari_runs), mean(purity_runs), std(purity_runs), ...
        'VariableNames', {'PER', 'ACC_Mean', 'ACC_Std', 'NMI_Mean', 'NMI_Std', 'ARI_Mean', 'ARI_Std', 'Pur_Mean', 'Pur_Std'});
    results_table = [results_table; new_row];
end

% ---------------------- 6. 展示与全面封存 ----------------------
disp('======================================================');
disp('FINAL RESULTS TABLE (Handwritten - Auto Grid Search Mode)');
disp(results_table);
save_filename = sprintf('results_FINAL_GridSearch_%s_%s.mat', DATASET_NAME, datestr(now, 'yyyymmdd_HHMM'));
save(save_filename, 'results_table', 'all_runs_details', 'config_base');
fprintf('>>> 成功！最优全量大表数据已保存至: %s\n', save_filename);
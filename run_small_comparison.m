% Small paired check. Not a paper-results reproduction.
revisionRoot=fileparts(mfilename('fullpath'));
addpath(revisionRoot,'-begin');clear main_fed_SR_FIMVC client_step_RWC;
cfg=struct('data_dir',fullfile(revisionRoot,'data'),'T',3,'anchor_m',30, ...
 'anchor_s',3,'kmeans_reps',1,'smart_axis',true,'metric','cosine', ...
 'R_anchor',4,'R_comm',3,'lambda1',1,'z_max_iter',20,'z_max_seconds',5,'z_constraint_tol',1e-7);
paired=struct([]);
for j=1:2
 rates=[0.3,0.5];cfg.PER=rates(j);cfg.structure_z_enabled=false;
 timer=tic;base=main_fed_SR_FIMVC('Handwritten',1,cfg);baseSeconds=toc(timer);
 cfg.structure_z_enabled=true;
 timer=tic;revised=main_fed_SR_FIMVC('Handwritten',1,cfg);revisedSeconds=toc(timer);
 assert(isequal(base.protocol.missing_rows,revised.protocol.missing_rows));
 assert(isequal(base.protocol.initial_assign,revised.protocol.initial_assign));
 paired(j).config=cfg;paired(j).base=base;paired(j).revised=revised;
 paired(j).base_seconds=baseSeconds;paired(j).revised_seconds=revisedSeconds;
 fprintf('PER %.1f FINAL: baseline ACC %.4f NMI %.4f ARI %.4f; revised ACC %.4f NMI %.4f ARI %.4f\n', ...
  cfg.PER,base.final.acc,base.final.nmi,base.final.ari,revised.final.acc,revised.final.nmi,revised.final.ari);
 save(fullfile(revisionRoot,'small_comparison.mat'),'paired');
end

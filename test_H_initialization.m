function test_H_initialization()
% Focused regression tests for rank-deficient latent initialization.
    here = fileparts(mfilename('fullpath'));
    addpath(here);

    rng(19);
    X = randn(80,6);
    [H, Hcluster, info] = initialize_latent_view(X,10);
    assert(info.data_rank == 6 && info.completed_columns == 4);
    assert(norm(H'*H-eye(10),'fro') < 1e-10);
    assert(norm(Hcluster(:,7:10),'fro') == 0);
    assert(norm(H(:,1:6)*H(:,1:6)' - ...
                Hcluster(:,1:6)*Hcluster(:,1:6)','fro') < 1e-10);

    [H2, Hcluster2, info2] = initialize_latent_view(X,10);
    assert(norm(H-H2,'fro') < 1e-10);
    assert(norm(Hcluster-Hcluster2,'fro') < 1e-10);
    assert(info2.orthogonality_error < 1e-10);

    [Hz, Hcz, infoz] = initialize_latent_view(zeros(30,4),8);
    assert(infoz.data_rank == 0 && infoz.completed_columns == 8);
    assert(norm(Hz'*Hz-eye(8),'fro') < 1e-10);
    assert(norm(Hcz,'fro') == 0);

    cfg = struct('T',1, 'PER',0.3, 'use_raw_init',false, ...
                 'structure_z_enabled',false, 'anchor_m',30, ...
                 'anchor_s',3, 'R_anchor',4, 'R_comm',3, ...
                 'kmeans_reps',1, 'data_dir',fullfile(here,'data'));
    R = main_fed_SR_FIMVC('Handwritten',1,cfg);
    init = R.protocol.H_initialization;
    assert(numel(init) == 6);
    orth_errors = cellfun(@(s)s.orthogonality_error, init);
    completed = cellfun(@(s)s.completed_columns, init);
    assert(max(orth_errors) < 1e-10);
    assert(completed(4) == 4);
    fprintf('PASS: H initialization; max orth error %.3g, view-4 completion %d.\n', ...
            max(orth_errors), completed(4));
end

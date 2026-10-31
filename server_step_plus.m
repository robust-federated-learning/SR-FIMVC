function server = server_step_plus(server, clients, config)
    V = numel(clients);
    k = config.k;
    % A paired cluster permutation leaves Y_v*C_v unchanged, so a separate
    % Hungarian alignment is algebraically redundant for the paper's Hc.
    Hc = cell(1,V);
    for v = 1:V
        Hc{v} = clients{v}.Y * clients{v}.C;
    end
    Hc = cat(2,Hc{:});
    [server.Y,server.C,server.update_info] = ...
        update_server_consensus(Hc,server.Y,server.C,config);

    % Refresh view-specific blocks sent back to clients.
    server.C_blocks = cell(1, V);
    for v = 1:V
        server.C_blocks{v} = server.C(:, (v-1)*k + (1:k));
    end
end

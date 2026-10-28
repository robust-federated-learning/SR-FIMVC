function result = apply_graph_operator(L,B)
% Apply a graph operator to an N-by-p matrix without forming N-by-N data.
% Numeric matrices remain supported for exact-reference tests and old callers.
    if isa(L,'function_handle')
        result=L(B);
    elseif isstruct(L) && isfield(L,'kind') && strcmp(L.kind,'consensus_laplacian')
        if size(B,1)~=numel(L.degree)
            error('SRFIMVC:GraphSizeMismatch','Graph and operand sample counts differ.');
        end
        result=bsxfun(@times,B,L.degree)-L.labels*(L.labels'*B);
    elseif isnumeric(L) || islogical(L)
        result=L*B;
    else
        error('SRFIMVC:UnsupportedGraph','Expected a matrix, consensus operator or function handle.');
    end
end

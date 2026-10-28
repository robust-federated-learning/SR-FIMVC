function M = masked_graph_operator(L,mask)
% M*B = G*L*G'*B, G=diag(mask). No diagonal or pairwise graph is formed.
    mask=double(mask(:));
    validateattributes(mask,{'numeric'},{'real','finite','nonnegative'});
    M=@(B) bsxfun(@times,mask, ...
        apply_graph_operator(L,bsxfun(@times,mask,B)));
end

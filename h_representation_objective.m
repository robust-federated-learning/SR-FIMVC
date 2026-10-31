function value=h_representation_objective(H,X,w,L,lambda_graph,align_target,lambda_align)
%H_REPRESENTATION_OBJECTIVE Evaluate the H objective stated in the paper.
    weighted_H=bsxfun(@times,H,w.^2);
    A_weighted_H=X*(X'*weighted_H)-lambda_graph*apply_graph_operator(L,weighted_H);
    value=sum(sum(H.*A_weighted_H))+lambda_align*sum(sum(H.*align_target));
    value=real(value);
end

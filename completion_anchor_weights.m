function [Z, anchors, meta] = completion_anchor_weights(X, m, config)
% Eq. (3): Euclidean nearest anchors and normalized Gaussian weights.
% Returns N-by-m assignments without constructing any N-by-N graph.
    [N,~] = size(X);
    if isfield(config,'anchor_s'), r=config.anchor_s; else, r=5; end
    validateattributes(r, {'numeric'}, {'real','finite','scalar','integer','positive'});
    r=min(r,m-1);
    [~,anchors]=lite_kmeans(X,m,'MaxIter',50,'Replicates',1,'Distance','sqEuclidean');
    D2=bsxfun(@plus,sum(X.^2,2),sum(anchors.^2,2)')-2*(X*anchors');
    D2=max(D2,0);
    [nearest,indices]=sort(D2,2,'ascend');
    nearest=nearest(:,1:r); indices=indices(:,1:r);
    if isfield(config,'anchor_sigma')
        validateattributes(config.anchor_sigma, {'numeric'}, {'real','finite','scalar','positive'});
        sigma2=config.anchor_sigma^2;
    else
        % Data-adaptive bandwidth; can be overridden by config.anchor_sigma.
        sigma2=mean(nearest(:,r));
        if ~isfinite(sigma2) || sigma2<=0, sigma2=1; end
    end
    % Subtract each row's minimum distance: cancels in the ratio and
    % prevents all Gaussian weights underflowing for a distant sample.
    weights=exp(-bsxfun(@minus,nearest,nearest(:,1))/(2*sigma2));
    weights=bsxfun(@rdivide,weights,sum(weights,2));
    rows=repmat((1:N)',1,r);
    Z=sparse(rows(:),indices(:),weights(:),N,m);
    meta=struct('r',r,'sigma2',sigma2,'metric','euclidean','anchors',anchors);
end

function [X_filled,means]=mean_fill_missing(X,Omega)
%MEAN_FILL_MISSING Fill only masked entries using observed column means.
% Missing values stored in X are never read, preventing hidden-truth leakage.
    if ~isequal(size(X),size(Omega)) || any(~isfinite(X(:))) || ...
            any(~isfinite(Omega(:))) || any(Omega(:)<0) || any(Omega(:)>1)
        error('mean_fill_missing:InvalidInput', ...
              'X and Omega must have equal size; Omega must be finite in [0,1].');
    end
    observed=Omega>0;
    counts=sum(observed,1);
    sums=sum(X.*observed,1);
    means=zeros(1,size(X,2));
    available=counts>0;
    means(available)=sums(available)./counts(available);
    X_filled=X.*observed+bsxfun(@times,~observed,means);
end

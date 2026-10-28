function assignment = custom_hungarian(costMatrix)
% =========================================================================


% 


%




% =========================================================================


    costMatrix(isnan(costMatrix)) = 1e6;
    


    try

        pairs = matchpairs(costMatrix, 1e6);
        

        num_clusters = size(costMatrix, 1);
        assignment = zeros(1, num_clusters);
        
        if ~isempty(pairs)

            assignment(pairs(:,1)) = pairs(:,2);
        end
        

        unassigned_rows = find(assignment == 0);
        unassigned_cols = setdiff(1:num_clusters, assignment);
        if ~isempty(unassigned_rows)
            assignment(unassigned_rows) = unassigned_cols(1:length(unassigned_rows));
        end
        
    catch

        [~, assignment] = max(-costMatrix, [], 2);

        if numel(unique(assignment)) < size(costMatrix, 1)
             assignment = 1:size(costMatrix, 1);
        end
    end
end
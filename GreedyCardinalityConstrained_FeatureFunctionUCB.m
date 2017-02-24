function [SummarySet] = GreedyCardinalityConstrained_FeatureFunctionUCB(feat_mat, feature_type, weights, Sigma, alpha, V, K, beta)
% takes the feature matrix with size N_V*N_Features, concave function type, 
% and the gamma, parameter associated with the concave function, alpha trades off between the exploitation and exploration
% weight vector weights, 

    remain_set = V;
    N_V = size(feat_mat, 1);
    N_F = size(feat_mat, 2);
    if N_V ~= length(V)
        fprintf('ERROR: the size of the input ground set is not the same as the data matrix');
    end
    
    SummarySet = [];
    precompute = zeros(1, N_F);
    for idx = 1:K
        replicated_Mat = (repmat(precompute,N_V, 1));
        augmented_Mat = feat_mat + replicated_Mat;  
        if feature_type == 1
            concave_replicated_Mat = replicated_Mat.^beta;
            concave_augmented_Mat = augmented_Mat.^beta;
            %updated_gains = ((feature_mat + replicated_Mat).^beta - replicated_Mat.^beta);
            %updated_gains = (feature_matrix.^beta - (repmat(precompute,N_V, 1)).^beta) * weights;
        elseif feature_type == 2
            concave_replicated_Mat = log(1+beta*replicated_Mat);
            concave_augmented_Mat = log (1+beta*augmented_Mat);
            %updated_gains = (log(1+beta*(feature_matrix + replicated_Mat)) - (log(1+beta*replicated_Mat))) * weights;
        elseif feature_type == 3
            concave_replicated_Mat = 1-beta.^(-replicated_Mat);
            concave_augmented_Mat = 1-beta.^(-augmented_Mat);
        elseif feature_type == 4 % 1/(1+exp^(-gamma * x)) - 0.5;            
            concave_replicated_Mat = 1./(1+exp(-beta*replicated_Mat)) - 0.5;
            concave_augmented_Mat = 1./(1+exp(-beta*augmented_Mat)) - 0.5;
        end
        var_mat = concave_augmented_Mat * Sigma * concave_augmented_Mat';
        std_vec = sqrt(diag(var_mat));
        %std_vec = (diag(var_mat));
        
        pred_vec = concave_augmented_Mat * weights;
        val_vec = pred_vec + alpha*std_vec;
        [max_val, max_item] = max(val_vec(remain_set));
        add_item = remain_set(max_item); 
        precompute = precompute + feat_mat(add_item, :);
        remain_set = setdiff(remain_set, add_item);
        SummarySet = [SummarySet, add_item];
    end
    
    %fprintf('Greedy algorithm outputs the performance: %f\n', Weighted_F_eval(F_vec, w_vec, SummarySet));
    
end
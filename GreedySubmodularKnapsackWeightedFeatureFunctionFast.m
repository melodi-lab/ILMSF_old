function [SummarySet] = GreedySubmodularKnapsackWeightedFeatureFunctionFast(feature_matrix, feature_type, weights, V, K, costList,alpha, beta)
%% greedy algorithm with feature based function only, requires the feature matrix, feature function type, and weights, K, V, costList, and alpha, the parameter associated with the concave function
% feature matrix is of size N*F, where F is the number of features
% feature type: 1, power function (x).^(beta), 2, log function, with the
% type log(1+beta*x)


remain_set = V;
N_V = length(V);
N_Features = size(feature_matrix, 2);
if length(costList) ~= length(V)
    fprintf('Error: size of costlist does not match with ground set size\n');
end
SummarySet = [];
precompute = zeros(1,N_Features);



while length(remain_set) > 0
    gain = -inf;
    add_item = -1;
    %prev_val = Weighted_F_eval(F_vec, w_vec, SummarySet);
    prev_val = 0;
    replicated_Mat = (repmat(precompute,N_V, 1));
    if feature_type == 1
        updated_gains = ((feature_matrix + replicated_Mat).^beta - replicated_Mat.^beta)*weights;
        %updated_gains = (feature_matrix.^beta - (repmat(precompute,N_V, 1)).^beta) * weights;
    elseif feature_type == 2
        updated_gains = (log(1+beta*(feature_matrix + replicated_Mat)) - (log(1+beta*replicated_Mat))) * weights;
    elseif feature_type == 3
        updated_gains = ((1-beta.^(-(feature_matrix+replicated_Mat))) - ((1-beta.^(-replicated_Mat)))) * weights;
    elseif feature_type == 4
        updated_gains = ((1./(1+exp(-beta*(feature_matrix+replicated_Mat))) - 0.5) - ((1./(1+exp(-beta*(replicated_Mat)) - 0.5)))) * weights;
    elseif feature_type == 5 % weighted maximum score
        updated_gains = (max(replicated_Mat, feature_matrix) - replicated_Mat) * weights;
    end
    normalized_gains = updated_gains./costList.^alpha;
    feasible_set = [];
    for idx = 1:length(remain_set)
        if sum(costList([SummarySet,remain_set(idx)])) <= K
            feasible_set = [feasible_set, idx];
        end       
    end
    remain_set = remain_set(feasible_set);
    if length(remain_set) == 0
        break;
    end
    [max_val, max_index] = max(normalized_gains(remain_set));
    add_item = remain_set(max_index);
    if add_item == -1
        break;
    end
    remain_set = setdiff(remain_set, add_item);
    SummarySet = [SummarySet, add_item];
    precompute = precompute + feature_matrix(add_item, :);
    %fprintf('iter %d, selecting %d with gain %f\n', length(SummarySet), add_item, normalized_gains(add_item));
end


%fprintf('Greedy algorithm outputs the performance: %f\n', Weighted_F_eval(F_vec, w_vec, SummarySet));
end
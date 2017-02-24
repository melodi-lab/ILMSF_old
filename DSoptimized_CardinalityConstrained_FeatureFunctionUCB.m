function [SummarySet] = DSoptimized_CardinalityConstrained_FeatureFunctionUCB(feat_mat, feature_type, weights, Sigma, alpha, V, K, beta)
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
%% initialize the solution with the randomized greedy algorithm

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
    [sorted_val_vec, sorted_val_order] = sort(val_vec(remain_set), 'descend');
    %max_item = sorted_val_order(randsample(length(sorted_val_vec),1,true, max(sorted_val_vec,0)));
    rand_order = randperm(K);
    max_item = sorted_val_order(rand_order(1));
    %[max_val, max_item] = max(val_vec(remain_set));
    add_item = remain_set(max_item);
    precompute = precompute + feat_mat(add_item, :);
    remain_set = setdiff(remain_set, add_item);
    SummarySet = [SummarySet, add_item];
end

if feature_type == 1
       concave_vec = precompute.^beta;
elseif feature_type == 2
        concave_vec = log(1+beta*precompute);
elseif feature_type == 3
        concave_vec = 1-beta.^(-precompute);
elseif feature_type == 4 % 1/(1+exp^(-gamma * x)) - 0.5;
        concave_vec = 1./(1+exp(-beta*precompute)) - 0.5;        
end

objective_value =  weights' * concave_vec' + alpha*sqrt(concave_vec*Sigma*concave_vec');


%% run the DS technique to optimize

last_objective_value = -inf;
newSigma = weights*weights'+alpha^2*Sigma; % compute the transformed Sigma
negNewSigma = newSigma;
negNewSigma(find(negNewSigma>0)) = 0; % truncate the Sigma to be negative
% find the modular upper bound of this function
absNegNewSigma = abs(negNewSigma); % make the matrix positive. 

round_idx = 1;

while last_objective_value < objective_value
    last_objective_value = objective_value; % store the solution of the last round
    lastSummarySet = SummarySet; % store the summary in the last round
    
    mod_vec = zeros(length(V),1); % initialize the modular function
    replicated_Mat = (repmat(precompute,N_V, 1));
    replicated_Empty_Mat = (repmat(zeros(size(precompute)),N_V, 1))+feat_mat; % for computing f(j) 
    reduced_Mat = replicated_Mat - feat_mat; % for computing f(j|S\j)
    if feature_type == 1
        concave_replicated_Mat = replicated_Mat.^beta;
        concave_augmented_Mat = reduced_Mat.^beta;
        concave_replicated_Mat_Empty = replicated_Empty_Mat.^beta;
        %updated_gains = ((feature_mat + replicated_Mat).^beta - replicated_Mat.^beta);
        %updated_gains = (feature_matrix.^beta - (repmat(precompute,N_V, 1)).^beta) * weights;
    elseif feature_type == 2
        concave_replicated_Mat = log(1+beta*replicated_Mat);
        concave_augmented_Mat = log (1+beta*reduced_Mat);
        concave_replicated_Mat_Empty = log(1+beta*replicated_Empty_Mat);
        %updated_gains = (log(1+beta*(feature_matrix + replicated_Mat)) - (log(1+beta*replicated_Mat))) * weights;
    elseif feature_type == 3
        concave_replicated_Mat = 1-beta.^(-replicated_Mat);
        concave_augmented_Mat = 1-beta.^(-reduced_Mat);
        concave_replicated_Mat_Empty = 1-beta.^(-replicated_Empty_Mat);
    elseif feature_type == 4 % 1/(1+exp^(-gamma * x)) - 0.5;
        concave_replicated_Mat = 1./(1+exp(-beta*replicated_Mat)) - 0.5;
        concave_augmented_Mat = 1./(1+exp(-beta*reduced_Mat)) - 0.5;
        concave_replicated_Mat_Empty = 1./(1+exp(-beta*replicated_Empty_Mat)) - 0.5;
    end
    result = concave_replicated_Mat * absNegNewSigma * concave_replicated_Mat' - concave_augmented_Mat * absNegNewSigma * concave_augmented_Mat';
    result_singleton = concave_replicated_Mat_Empty * absNegNewSigma * concave_replicated_Mat_Empty';
    for idx = 1:length(V)
        if length(find(lastSummarySet == idx))==1 % is in the summary
            mod_vec(idx) = result(idx);
        else % not in the summary
            mod_vec(idx) = result_singleton(idx);
        end
    end
    
    SummarySet = [];
    precompute = zeros(1, N_F);
    % run the greedy algorithm
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
        val_vec = pred_vec + alpha*std_vec - sqrt(mod_vec);
        [sorted_val_vec, sorted_val_order] = sort(val_vec(remain_set), 'descend');
        %max_item = sorted_val_order(randsample(length(sorted_val_vec),1,true, max(sorted_val_vec,0)));
        rand_order = randperm(K);
        max_item = sorted_val_order(rand_order(1));
        %[max_val, max_item] = max(val_vec(remain_set));
        add_item = remain_set(max_item);
        precompute = precompute + feat_mat(add_item, :);
        remain_set = setdiff(remain_set, add_item);
        SummarySet = [SummarySet, add_item];
    end
    
    if feature_type == 1
       concave_vec = precompute.^beta;
    elseif feature_type == 2
        concave_vec = log(1+beta*precompute);
    elseif feature_type == 3
        concave_vec = 1-beta.^(-precompute);
    elseif feature_type == 4 % 1/(1+exp^(-gamma * x)) - 0.5;
        concave_vec = 1./(1+exp(-beta*precompute)) - 0.5;        
    end

    objective_value =  weights' * concave_vec' + alpha*sqrt(concave_vec*Sigma*concave_vec');
    
    fprintf('DS iteration: %d\n', round_idx);
    round_idx = round_idx + 1;
end

SummarySet = lastSummarySet;
end
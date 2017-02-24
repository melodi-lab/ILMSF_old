function create_OCS_results(N_samples, feature_based_func_type, gamma, noise_level, threshold_zero_flag, feedback_model, verbose, N_bins, alpha)
%% this script runs the learning scenario where the tradeoff between exploration and exploitation is leveraged. (OCS algorithm)

%% Following is the description of each parameter:
% N_samples: number of rounds in the interactive experiment
% feature_based_func_type: the type of the concave function you want to
% use. 1: x.^gamma with 0 < gamma < 1; 2 log (1+gamma*x) with gamma >0; 3: 1 - gamma^(-x) with gamma > 1; 4: 1/(1+exp^(-gamma * x)) - 0.5;
% gamma: the parameter defining the concave function. 
% noise_level: the parameter associated with the linear regression problem.
% threshold_zero_flag: 1: project the learned parameter back to positive
% orthant, 0: don't project the learned parameter back to positive orthant.
% the performance is not very sensitive to this parameter.
% feedback_model: always setting it as 1. This parameter is not necessary
% in the user study experiments.
% verbose: set any value for this parameter. It is ineffective in the user
% study.
% N_bins: this parameter is ineffective when feedback_model=1. Ignore this
% parameter in the user study experiments.
% alpha: the tradeoff parameter between exploitation and exploration.
% Larger alpha implies more on exploration.

%%

N_ground = 100; % this number should be fixed in this case
N_dim = 628; % this number should also be fixed
K_budget = 10; % this number should be fixed
data_dir = 'data_folder/';

load(['processed_data.mat']);

load(['preprocessed_random_summaries_and_scores.mat']);


mu_0 = zeros(N_dim,1);
sigma_0 = eye(N_dim);
method = 'uncertainty_alpha'; % random or uncertainty


if feedback_model == 1
    rand_result_file = ['RandRes_Nsmps_', num2str(N_samples), '_FeaFncTp_', num2str(feature_based_func_type), '_ga_', num2str(gamma),'_NsLv_', num2str(noise_level), ...
    '_Thr0Flg_', num2str(threshold_zero_flag),'_FdMdl_', num2str(feedback_model),'.mat'];
    uncertainty_greedy_result_file = ['UnctyGrdyAlpha_Res_Nsmps_', num2str(N_samples), '_FeaFncTp_', num2str(feature_based_func_type), '_ga_', num2str(gamma),'_NsLv_', num2str(noise_level), ...
    '_Thr0Flg_', num2str(threshold_zero_flag),'_FdMdl_', num2str(feedback_model), '_Alfa_', num2str(alpha)  ,'.mat'];
end

if exist([data_dir, uncertainty_greedy_result_file], 'file') == 2
    fprintf('The experiment has been run, and move on\n');
    return ;
end

clear Collected_Random_Result_Mat;
clear Collected_Uncertainty_Result_Mat;
clear Collected_Uncertainty_Queried_Summaries;
clear Collected_Uncertainty_Queried_Rouge_Score;
load([data_dir, rand_result_file]);

figure();
for idx = 1:14
    subplot(5,3,idx);
    Data_Mat = all_Feature_Vec{idx}';
    X_mat = [];
    Y = [];
    sigma = sigma_0;
    perf_vec = zeros(N_samples,1);
    Uncertainty_Queried_Summaries = zeros(K_budget, N_samples);
    Uncertainty_Queried_Rouge_Score = zeros(1, N_samples);
    for jdx = 1:N_samples
        if strcmp(method, 'uncertainty_alpha') == 1
            if jdx == 1
                w_vec = ones(N_dim,1);
            end
            %alpha = .1;
            SummarySet = DSoptimized_CardinalityConstrained_FeatureFunctionUCB(Data_Mat', feature_based_func_type, w_vec, (sigma), alpha, V, K, gamma); 
            Uncertainty_Queried_Summaries(:, jdx) = SummarySet';
        end
        x = Featurize_data(Data_Mat, feature_based_func_type, gamma, SummarySet);
        y = score_image_summarization(all_subset{idx}, all_Feature_Vec{idx}, SummarySet);
        real_val = y;
        
        X_mat = [X_mat;x'];
        Y = [Y;y];

        [w_vec, sigma] = Linear_Regression_Learning(X_mat, Y, mu_0, sigma_0, noise_level);
        if threshold_zero_flag == 1
            w_vec = max(w_vec, 0);
        end
        optimized_summary = GreedySubmodularKnapsackWeightedFeatureFunctionFast(Data_Mat', feature_based_func_type, w_vec, V, K_budget, ones(N_ground,1), 1, gamma);
        optimized_y = score_image_summarization(all_subset{idx}, all_Feature_Vec{idx}, optimized_summary);
        if verbose > 0
            fprintf('Collection %d: sample id = %d, avg_random=%f, avg_human=%f, best_performance=%f, learned_performance=%f, queried_feedback=%f,%f\n', ...
                idx, jdx, mean(all_random_scores{idx}), mean(all_human_score{idx}), V_rouge_optimized_summary_score{idx}, optimized_y, y, real_val);
        end
        perf_vec(jdx) = optimized_y;
    end
    plot(perf_vec, 'r-'); hold on;
    Collected_Uncertainty_Result_Mat{idx} = perf_vec;
    Collected_Uncertainty_Queried_Summaries{idx} = Uncertainty_Queried_Summaries;
    Collected_Uncertainty_Queried_Rouge_Score{idx} = Uncertainty_Queried_Rouge_Score;
    plot(mean(Collected_Random_Result_Mat{idx}), 'b-'); hold on;
    plot(mean(Collected_Random_Result_Mat{idx}) + std(Collected_Random_Result_Mat{idx}), 'b--'); hold on;
end
title(uncertainty_greedy_result_file);
hold off;



save([data_dir, uncertainty_greedy_result_file], 'Collected_Uncertainty_Result_Mat', 'Collected_Uncertainty_Queried_Summaries', 'Collected_Uncertainty_Queried_Rouge_Score');

end
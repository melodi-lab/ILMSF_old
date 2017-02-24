function create_random_results_baseline(N_samples, feature_based_func_type, gamma, noise_level, threshold_zero_flag, feedback_model, verbose, N_bins)
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

%% set up parameters, this script process all image collections, in particular collect the human score, the features representation, and random summaries scores


N_ground = 100; % this number should be fixed in this case
N_dim = 628; % this number should also be fixed
K_budget = 10; % this number should be fixed
data_dir = 'data_folder/';

load(['processed_data.mat']); % load the processed results for fast experimental turnaround. 
% If the .mat file cannot be read, try generating the features yourself
% using the script "image_collection_preprocessing.m"

load(['preprocessed_random_summaries_and_scores.mat']); % load the processed scores and random results.
% similar to the above, if the .mat file cannot be read, try generating the
% scores yourself using the script "image_collection_preprocessing.m". 


%% create the random results and store them


mu_0 = zeros(N_dim,1); % initialize the mean
sigma_0 = eye(N_dim); % initialize the covariance matrix
N_random_runs = 10; % number of random runs 



if feedback_model==1
    output_file_name = ['RandRes_Nsmps_', num2str(N_samples), ...
    '_FeaFncTp_', num2str(feature_based_func_type), '_ga_', num2str(gamma), '_NsLv_', num2str(noise_level), ...
    '_Thr0Flg_', num2str(threshold_zero_flag), '_FdMdl_', num2str(feedback_model), '.mat'];
end


if exist([data_dir, output_file_name], 'file') == 2
    fprintf('The experiment has been run, and move on\n');
    return ;
end


clear Collected_Random_Result_Mat;

tic;
for idx = 1:14
    Data_Mat = all_Feature_Vec{idx}';
    Random_Result_Mat = zeros(N_random_runs, N_samples);

    for random_idx = 1:N_random_runs
        X_mat = [];
        Y = [];
        perf_vec = zeros(1,N_samples);
        cnt = 1;
        for jdx = ((random_idx-1) * N_samples + 1) : (random_idx * N_samples)
            %jdx
            SummarySet = Collected_All_Random_Summaries{idx}(:, jdx);
            x = Featurize_data(Data_Mat, feature_based_func_type, gamma, SummarySet);
            y = Collected_All_Random_Rouge_Scores{idx}(jdx);

            X_mat = [X_mat;x'];
            Y = [Y;y];
            w_vec = Linear_Regression_Learning(X_mat, Y, mu_0, sigma_0, noise_level);
            if threshold_zero_flag == 1
                w_vec = max(w_vec, 0);
            end
            % run the greedy algorithm to select a summary
            optimized_summary = GreedySubmodularKnapsackWeightedFeatureFunctionFast(Data_Mat', feature_based_func_type, w_vec, V, K_budget, ones(N_ground,1), 1, gamma); 
            optimized_y = score_image_summarization(all_subset{idx}, all_Feature_Vec{idx}, optimized_summary);
            if verbose > 0
                fprintf('Collection %d: random run = %d, sample = %d, avg_random=%f, avg_human=%f, best_performance=%f, learned_performance=%f, queried_feedback=%f\n', ...
                idx, random_idx, jdx, mean(all_random_scores{idx}), mean(all_human_score{idx}), V_rouge_optimized_summary_score{idx}, optimized_y, y);
            end
            perf_vec(cnt) = optimized_y;
            cnt = cnt + 1;
        end
        Random_Result_Mat(random_idx, :) = perf_vec;
    end
    Collected_Random_Result_Mat{idx} = Random_Result_Mat;
end

toc;


save([data_dir, output_file_name], 'Collected_Random_Result_Mat'); % store the running results.

end
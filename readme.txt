In this folder, we include all the matlab scripts needed for simulating the interactive learning of mixtures of submodular functions on the image collection summarization data set.

-----------


To simulate the case where random summaries are chosen for labeling, you could run the following matlab script:

create_random_results_baseline(N_samples, feature_based_func_type, gamma, noise_level, threshold_zero_flag, feedback_model, verbose, N_bins)


Details about the parameters:


N_samples: number of rounds in the interactive experiment

feature_based_func_type: the type of the concave function you want to use.
	1: x.^gamma with 0 < gamma < 1;
	2 log (1+gamma*x) with gamma >0;
	3: 1 - gamma^(-x) with gamma > 1;
	4: 1/(1+exp^(-gamma * x)) - 0.5;
(I suggest setting the concave function type as 2, i.e., log(1+gamma*x))

gamma: the parameter defining the concave function.
(If concave function type is chosen as 2, I suggest setting gamma as 100)

noise_level: the parameter associated with the linear regression problem. (I suggest setting it as 0.1)

threshold_zero_flag:
	1: project the learned parameter back to positive orthant,
	0: don't project the learned parameter back to positive orthant.
(Suggest setting it as 1, but the performance is not very sensitive to this parameter).

feedback_model: always setting it as 1.
This parameter is not necessary in the user study experiments.

verbose: set any value for this parameter. It is ineffective in the user study.

N_bins: this parameter is ineffective when feedback_model=1. Ignore this parameter in the user study experiments.

-----------



To simulate the case where summaries are chosen by greedily optimizing the uncertainty sample objective, you could run the following script:

create_uncertainty_greedy_results(N_samples, feature_based_func_type, gamma, noise_level, threshold_zero_flag, feedback_model, verbose, N_bins)



The parameters in this script are the same as the previous script.

-----------


To simulate the case where summaries are chosen by optimizing the uncertainty sampling objective using DS techniques, you could run the following script:


create_uncertainty_DS_optimization_results(N_samples, feature_based_func_type, gamma, noise_level, threshold_zero_flag, feedback_model, verbose, N_bins)



The parameters in this script are the same as the previous script.

-----------


To simulate the case where summaries are chosen by optimizing the online learning algorithm (OCS in the paper), you could run the following script:


create_OCS_results(N_samples, feature_based_func_type, gamma, noise_level, threshold_zero_flag, feedback_model, verbose, N_bins, alpha)


alpha controls the trade off between exploitation and exploration.


===========================
Code files
===========================

Building blocks : (to put in funcs.py)
	featurize_data
	linear_regression_learning
	score_image_summarization
	image_collection_preprocessing (??) : uses score_image_summ.

Functions Returning Summaries :
	DSoptimized_CardinalityConstrained_FeatureFunctionUCB
	GreedyCardinalityConstrained_FeatureFunctionUCB
	GreedySubmodularKnapsackWeightedFeatureFunctionFast

create results : (using above summaries)
	create_OCS_results
	create_random_results_baseline
	create_uncertainty_DS_optimization_results
	create_uncertainty_greedy_results


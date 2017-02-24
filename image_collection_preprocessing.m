function [human_score, Feature_Vec, subset, random_scores] = image_collection_preprocessing(imagecollection)

%imagecollection = 8; % collection id
N_images = 100; % this number should be fixed in this case
N_dim = 628; % this number should also be fixed
N_random_samples = 500; % number of random samples
K_budget = 10; % this number should be fixed
Feature_Vec = zeros(N_images, N_dim);
%addpath('~/projects/matlab_script/');

% read feature file
for idx = 1:N_images
    fid = fopen(['feature_data/set', num2str(imagecollection), '/img', num2str(idx), '.vec'], 'rb');
    a = fread(fid, 'float');
    Feature_Vec(idx,:) = a;
    fclose(fid);
end

%Feature_Vec(:, 201) = 0;
% read the summaries

% read human summaries
fid = fopen(['summaries/set', num2str(imagecollection) , '/summary_0.05.txt'], 'rt');
%fid = fopen(['summaries/set', num2str(imagecollection) , '/summary_0.10.txt'], 'rt');
%fid = fopen(['summaries/set', num2str(imagecollection) , '/summary.txt'], 'rt');


index = 1;
human_score = [];
clear subset;
while ~feof(fid)
    line = fgetl(fid);
    subset{index} = sscanf(line, '%d ');
    %score_image_summarization(subset, Feature_Vec, subset{index});
    index = index + 1;
end
fclose(fid);
for idx = 1:length(subset)
    human_score = [human_score, score_image_summarization_F_measure(subset, Feature_Vec, subset{idx})];
end

random_scores = [];
for idx = 1:N_random_samples
    rand_set = randperm(N_images);
    rand_set = rand_set(1:K_budget);
    %[dummy, y] = score_image_summarization(subset, Feature_Vec, rand_set);
    random_scores = [random_scores, score_image_summarization(subset, Feature_Vec, rand_set)];
end

end
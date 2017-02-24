function [score, scale_val]= score_image_summarization(subset, Feature_Vec, eval_summary)
    score = 0;
    
    val_vec = zeros(length(subset),size(Feature_Vec,2));
    eval_val_vec = sum(Feature_Vec(eval_summary,:));
    for idx = 1:length(subset)
        val_vec(idx,:) = sum(Feature_Vec(subset{idx}, :));
        score = score + sum(min(val_vec(idx,:), eval_val_vec));
    end
    scale_val = score;
    score = score / sum(val_vec(:));
    %eval_val_vec = zeros(1, size(Feature_Vec,2));   
    %sum(min(x, val_vec))
    
end
function x = Featurize_data(Data_Mat, feature_based_func_type, gamma, SummarySet)
    if feature_based_func_type == 1 % x.^gamma with 0 < gamma < 1 
        x = sum(Data_Mat(:, SummarySet),2).^gamma;
    elseif feature_based_func_type == 2 % log (1+gamma*x) with gamma >0
        x = log(1+gamma*sum(Data_Mat(:,SummarySet),2));
    elseif feature_based_func_type == 3 % 1 - gamma^(-x) with gamma > 1
        x = 1-gamma.^(-sum(Data_Mat(:,SummarySet), 2));
    elseif feature_based_func_type == 4 % 1/(1+exp^(-gamma * x)) - 0.5;
        x = 1./(1+exp(-gamma*sum(Data_Mat(:,SummarySet), 2))) - 0.5;
    end
end
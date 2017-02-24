function [mu, sigma] = Linear_Regression_Learning(X_mat, Y, mu_0, sigma_0, noise_level)
    sigma = inv(inv(sigma_0) + 1/(noise_level^2) * X_mat' * X_mat);
    %sigma = inv(scale_normalization*inv(sigma_0) + X_mat' * X_mat);
    mu = sigma * (inv(sigma_0) * mu_0 + 1/(noise_level^2) * X_mat' * Y);
    %mu = sigma * (X_mat' * Y);
    %w_vec = mu(1:n_func);
    %sigma = sigma(1:n_func, 1:n_func);

end
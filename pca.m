function [U, S] = pca(X)
% PCA for large matrices.
% Runs principal component analysis on the dataset X
% [U, S] = pca(X) computes eigenvectors of the covariance matrix of X
% Returns the eigenvectors U, the eigenvalues (on diagonal) in S
% Useful values
[m, n] = size(X);

U = zeros(n);
S = zeros(n);

eps = (X' * X) ./ m;
[U,S,~] = svd(eps);
end

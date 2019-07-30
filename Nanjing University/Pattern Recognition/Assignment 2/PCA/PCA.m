clc;clear all; 
% generate data and plot before PCA
x = randn(2000, 2) * [2 1; 1 2];
%figure;
scatter(x(:, 1),x(:, 2));
hold on
% perform PCA and plot
x_mean = mean(x); % 1*2
x_new = zeros(2000, 2);
for i = 1:2000
    x_new(i, :) = x(i, :) - x_mean;
end
Covx = cov(x_new); % 2 * 2
[V, D] = eig(Covx); % D:2 * 2
d = zeros(1, 2);
if 0  % same as diag()
for i = 1:2
   d(1, i) = D(i, i); 
end
end
d = (diag(D))'; % 1 * 2
[D_sort, index] = sort(d, 'descend');
if 0 % same as V(:, index)
for i = 1:2
    V_sort(:, i) = V(:, index(1,i));
end
end
V_sort = V(:, index);

x_pca = x_new * V_sort';
%figure;
scatter(x_pca(:, 1),x_pca(:, 2));

% perform whitening operation and plot
x_pca_whiten = x_pca * diag(1./sqrt(d));
%figure;
scatter(x_pca_whiten(:, 1),x_pca_whiten(:, 2));




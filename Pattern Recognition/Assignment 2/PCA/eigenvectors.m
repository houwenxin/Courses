clear all;clc;
% set the random number seed to 0 for reproducibility.
%rand('seed', 0);
rng(0);
avg = [1 2 3 4 5 6 7 8 9 10];
scale = 0.1;

%generate 5000 examples, each 10 dim
data = randn(5000, 10) + repmat(avg*scale, 5000, 1);

m = mean(data); % average 1*10
m1 = m / norm(m); % normalized average
%do PCA, but without centering
[~, S, V] = svd(data);
S = diag(S);
e1 = V(:, 1); %first eigenvector, not minus mean vector

%do correct PCA with centering
newdata = data - repmat(m, 5000, 1);
[U, S, V] = svd(newdata);
S = diag(S);
new_e1 = V(:, 1); % first eigenvector, minus mean vector

% correlation between first eigenvector ( new & old ) and mean
avg = avg - mean(avg);
avg = avg / norm(avg);
e1 = e1 - mean(e1);
e1 = e1 / norm(e1);
new_e1 = new_e1 - mean(new_e1);
new_e1 = new_e1 / norm(new_e1);
corr1 = avg * e1
corr2 = e1' * new_e1
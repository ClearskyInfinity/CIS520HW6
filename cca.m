clear;
clc;

data_dir = 'data';
filename = 'breast_cancer.mat';
filepath = sprintf('%s/%s', data_dir, filename);
breast_cancer_data = load(filepath);
X = breast_cancer_data.X_train;
Y = breast_cancer_data.Y_train * 1.0;

L = (X' * X) ^ (-1/2);
R = (Y' * Y) ^ (-1/2);
M = X' * Y;

Z = L * M * R;

[U, S, V] = svd(Z);

XU = X * U;
YV_t = Y * V';

corr_cca = max(corr(XU, YV_t));

x_pca = pca(X);
z = X * x_pca(:,1);
b = z\Y;
y_hat = b * z;

corr_pcr = corr(y_hat, Y);
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

disp(U);

XU = X * U;
YV_t = Y * V';

correlation = corr(XU, YV_t);

disp(correlation);

y = Y;










clear;
clc;

%% load data

load ./data/ocr_train.mat
load ./data/ocr_test.mat

%% pca

[score_train, score_test, numpc] = pca_getpc(X_train, X_test);

% your code to select new features using PCA-ed data
x_train_pca = score_train(:, 1:numpc);
x_test_pca = score_test(:, 1:numpc);

%% auto encoder

addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
 
% your code to train an Auto-encoder, then learn new features from the original data set
% use rbm.m and newFeature_rbm.m
dbn = rbm(X_train);
[x_train_ae, x_test_ae] = newFeature_rbm(dbn, X_train, X_test);

%% logistic

addpath('./liblinear');
% precision_ori_log = logistic(X_train, Y_train, X_test, Y_test);
% 
% % your code to train logistic on PCA-ed and Auto-encoder data
% precision_pca_log = logistic(x_train_pca, Y_train, x_test_pca, Y_test);
% precision_ae_log  = logistic(x_train_ae, Y_train, x_test_ae, Y_test);

%% kmeans

K = [26, 50];
precision_ori_km = zeros(length(K), 1);
precision_pca_km = zeros(length(K), 1);
precision_ae_km  = zeros(length(K), 1);

for i = 1:length(K)
    k = K(i);
    precision_ori_km(i) = k_means(X_train, Y_train, X_test, Y_test, k);
    precision_pca_km(i) = k_means(x_train_pca, Y_train, x_test_pca, Y_test, k);
    precision_ae_km(i)  = k_means(x_train_ae, Y_train, x_test_ae, Y_test, k);
end

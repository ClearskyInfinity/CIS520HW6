clear;
clc;
data_dir = 'data';
digit_train_file = 'MNIST_train.mat';
digit_test_file  = 'MNIST_test.mat';
train_file_path = sprintf('%s/%s', data_dir, digit_train_file);
test_file_path  = sprintf('%s/%s', data_dir, digit_test_file);

tmp = load(train_file_path);
Xtrain = tmp.X_train;
Ytrain = tmp.Y_train;

tmp = load(test_file_path);
Xtest = tmp.X_test;
Ytest = tmp.Y_test;

[m, n] = size(Xtrain);
x_bar = mean(Xtrain, 1);
X_standardized = Xtrain - repmat(x_bar, [m, 1]);
X_test_stand = Xtest - repmat(x_bar, [size(Xtest, 1), 1]);
[coeff, score, ~] = pca(X_standardized);

%% Computation for part a)
% pc_components = score(:, 1 : 2);
% zero_indices = find(Ytrain == 1);
% one_indices = find(Ytrain == 2);
% zero_pcs = pc_components(zero_indices, :);
% one_pcs = pc_components(one_indices,  :);
% plot(zero_pcs(:, 1), zero_pcs(:, 2), 'o');
% title('Plot of 0-1 digits from top 2 PCA dimensions');
% xlabel('PC1');
% ylabel('PC2');
% hold on;
% plot(one_pcs(:, 1), one_pcs(:, 2), 'x');
% legend('0 digit', '1 digit');
% hold off;

%% Compute reconstruction error. Part b)
% as a matter of fact, this is exactly how we program.
% x_bar_rep = repmat(x_bar, [m 1]);
% accuracy = zeros(n, 1);
% loadings = coeff';
% 
% for k = 1 : n
% %     k = 147;
%     x_hat = x_bar_rep + score(:, 1:k) * loadings(1:k, :);
%     
%     diff1 = x_hat - x_bar_rep;
%     diff2 = Xtrain - x_bar_rep;
%     accuracy(k) = norm(diff1, 'fro')^2 / norm(diff2, 'fro')^2;
%     disp(accuracy(k));
% end
% plot(error);
% xlabel('Number of Principal Components');
% ylabel('Reconstruction Accuracy');

%% Part c)
num_clusters = 10;
k = [100, 150, 200];
precision = zeros(1, size(k, 2));

for i = 1 : size(k, 2)
    x_train_pca = X_standardized * coeff(:, 1:k(i));
    x_test_pca = X_test_stand * coeff(:, 1:k(i));
    precision(i) = k_means(x_train_pca, Ytrain, x_test_pca, Ytest, num_clusters);
end

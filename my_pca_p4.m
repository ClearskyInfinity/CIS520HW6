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
[coeff, score, ~] = pca(X_standardized);

% Computation for part a)
% pc_components = score(:, 1 : 2);
% zero_indices = find(Ytrain == 1);
% one_indices = find(Ytrain == 2);
% zero_pcs = pc_components(zero_indices, :);
% one_pcs  = pc_components(one_indices,  :);
% plot(zero_pcs(:, 1), zero_pcs(:, 2), 'o');
% title('Plot of 0-1 digits from top 2 PCA dimensions');
% xlabel('PC1');
% ylabel('PC2');
% hold on;
% plot(one_pcs(:, 1), one_pcs(:, 2), 'x');
% legend('0 digit', '1 digit');
% hold off;

% Compute reconstruction error. Part b)
% as a matter of fact, this is exactly how we program.
x_bar_rep = repmat(x_bar, [m 1]);
error = zeros(n, 1);
loadings = coeff';

for k = 1 : n
%     k = 616;
    x_hat = x_bar_rep + score(:, 1:k) * loadings(1:k, :);
    
    diff1 = x_hat - x_bar_rep;
    distortion1 = 0;
    for i = 1 : length(diff1)
        distortion1 = distortion1 + norm(diff1(i), 2)^2;
    end
    
    diff2 = Xtrain - x_bar_rep;
    distortion2 = 0;
    for i = 1 : length(diff2)
        distortion2 = distortion2 + norm(diff2(i), 2)^2;
    end
    
    error(k) = distortion1 / distortion2;
%     disp(error(k));
end
plot(error);
xlabel('Number of Principal Components');
ylabel('Reconstruction Accuracy');

% Part c)
% num_clusters = 10;
% 
% K1 = 100;
% K2 = 150;
% K3 = 200;
% 
% [IDX, C] = kmeans(score(:, 1 : K3), num_clusters);
% cluster_digits = zeros(num_clusters, 1);
% 
% for cluster = 1 : num_clusters
%     cluster_indices = find(IDX == cluster);
%     cluster_labels = Ytrain(cluster_indices);
%     cluster_digits(cluster) = mode(cluster_labels);
% end
% [m_test, n_test] = size(Xtest);
% predictions = zeros(m_test, 1);
% X_test_stand = Xtest - repmat(x_bar, [m_test, 1]);
% PC_loadings = coeff(:, 1: K3);
% reduced_Xtest = X_test_stand * PC_loadings;
% for i = 1 : m_test
%     dist = zeros(num_clusters, 1);
%     for cluster = 1 : num_clusters
%         dist(cluster) = norm(reduced_Xtest(i) - C(cluster), 2);
%     end
%     [~, index] = min(dist);
%     predictions(i) = cluster_digits(index);
% end
% 
% a = find(predictions ~= Ytest);
% disp(size(a));

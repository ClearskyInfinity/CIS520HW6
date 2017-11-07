data_dir = 'data';
digit_train_file = 'MNIST_train.mat';
digit_test_file = 'MNIST_test.mat';

train_file_path = sprintf('%s/%s', data_dir, digit_train_file);
test_file_path  = sprintf('%s/%s', data_dir, digit_test_file);

tmp = load(train_file_path);
Xtrain = tmp.X_train;
Ytrain = tmp.Y_train;
[coeff, score, latent] = pca(Xtrain);

%% Compute plot for part a)

% pc_components = score(:, 1 : 2);
% zero_indices = find(Ytrain == 1);
% one_indices   = find(Ytrain == 2);
% zero_pcs = pc_components(zero_indices, :);
% one_pcs  = pc_components(one_indices,  :);
% plot(zero_pcs(:, 1), zero_pcs(:, 2), 'ro');
% title('Plot of 0-1 digits from top 2 PCA dimensions');
% xlabel('PC1');
% ylabel('PC2');
% hold on;
% plot(one_pcs(:, 1), one_pcs(:, 2), 'bs');
% hold off;

%% Compute reconstruction error.
x_bar = mean(Xtrain, 1);
% as a matter of fact, this is exactly how we program.

disp(x_bar);

%% 

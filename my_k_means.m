clear;
clc;

p  = cell(1, 1);
p{1}  = [2,  1];
p{2}  = [3,  1];
p{3}  = [2,  2];
p{4}  = [3,  2];
p{5}  = [8,  9];
p{6}  = [7,  8];
p{7}  = [7,  7];
p{8}  = [8,  6];
p{9}  = [12, 9];
p{10} = [13, 8];
p{11} = [13, 7];
p{12} = [12, 6];
p{13} = [17, 3];
p{14} = [17, 2];
p{15} = [17, 1];

init_mu1 = [[7, 7];[8, 9];[12, 9]];
init_mu2 = [[12,6];[8, 9];[12, 9]];

centroid_legends = {'bo', 'rs', 'g:^'};
cluster_colors = {'b', 'r', 'g'};
paint_data_points(p, 1, []);
paint_centroids(init_mu2, centroid_legends);

N = length(p); % N is the number of sample points.

mu = init_mu2;

for i = 2 : 5
    assignment = assign_clusters(p, mu);
    paint_data_points(p, i, assignment);
    mu = set_centroid(p, assignment, init_mu1);
    paint_centroids(mu, centroid_legends);
end

function [] = paint_data_points(p, i, assignments) 
    tmp = cat(1, p{:});
    figure(i);
    if length(assignments) == 0
        plot(tmp(:, 1), tmp(:, 2), 'k*');    
    else
        hold on;
        for j = 1 : length(tmp)
            point = tmp(j, :);
            if assignments(j) == 1
                plot(point(1), point(2), 'b*');
            elseif assignments(j) == 2
                plot(point(1), point(2), 'r*');
            else
                plot(point(1), point(2), 'g*');
            end
        end        
        hold off;
    end
    axis equal;
    set(gca, 'XLim',  [0, 20]);
    set(gca, 'XTick', 1.0 : 20.0);
    set(gca, 'YLim',  [0, 10]);
    set(gca, 'YTick', 1.0 : 10.0);
    grid on;
end

function [] = paint_centroids(mu, centroid_legends)
    hold on;
    for i = 1 : length(mu)
        plot(mu(i, 1), mu(i, 2), centroid_legends{i}, 'markers', 12);
        text(mu(i, 1),mu(i, 2),['(' num2str(mu(i, 1)) ',' num2str(mu(i, 2)) ')'])
    end
    hold off;
end

function [assignments] = assign_rand_centroids(num_clusters, num_centroids)
    random_sequence = randperm(num_clusters);
    assignments = mod(random_sequence, num_centroids) + 1;
end

function [assignment] = assign_clusters(p, mu)    
    assignment = zeros(1, length(p));
    for i = 1 : length(p)
        point = p{i};
        dist = zeros(1, 3);
        for j = 1 : length(mu)
            tmp = mu(j, :);
            d = [ point(1) - tmp(1), point(2) - tmp(2)];
            dist(j) = norm(d, 2)^2;
        end
        [~, closest_cluster_index] = min(dist);
        assignment(i) = closest_cluster_index;
    end
end

function [updated_mu] = set_centroid(p, assignments, mu)
    updated_mu = zeros(3, 2);
    for k = 1 : length(mu)
        indices = find(assignments == k);
        updated_mu(k, :) = sum(cat(1, p{indices}),1) / length(indices);
    end
end


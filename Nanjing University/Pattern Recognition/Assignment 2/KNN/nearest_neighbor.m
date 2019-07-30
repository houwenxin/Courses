function [indice, min_distance] = nearest_neighbor(test_data, dataset)

[n, m] = size(dataset);
distance = zeros(1, m);
for i = 1:m
	distance(1, i) = norm(dataset(:, i)-test_data);
	%fprintf("Distance 1: %d\n",norm(dataset(:,i)-test_data));
	%fprintf("Distance 2: %d\n\n",sqrt((sum(dataset(:, i)-test_data).^2)));
end
[distances, indices] = sort(distance);
min_distance = distances(1);
indice = indices(1);
%fprintf("Size of distance matrix: %d %d\n",size(distance));
end


function [min_distance, indice] = nearest_neighbor(test_data, dataset)

[m, n] = size(dataset);
distance = zeros(m);
for i = 1:m
	distance(i) = sqrt(sum((dataset(i,:)-test_data).^2));
end
[distances, indices] = sort(distance);
min_distance = distances(1);
indice = indices(1);
end


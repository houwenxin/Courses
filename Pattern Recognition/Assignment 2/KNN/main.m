clear; close all; clc;
data = generate_data();
[m, n] = size(data);
result = zeros(m,2);

for i = 1:m
	[result(i,1), result(i,2)] = nearest_neighbor(data(i, :), [data(1:i-1, :); data(i+1:m, :)]);
end
disp(result)



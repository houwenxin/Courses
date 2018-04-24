clear; close all; clc;
data = csvread('dataset.txt');
[n, m] = size(data);
result = zeros(m,2);
result_2 = zeros(m,2); % Predicted by VLfeat Kdtrees
t1 = clock;
for i = 1:m
	[result(i,1), result(i,2)] = nearest_neighbor(data(:, i), [data(:, 1:i-1) data(:, i+1:m)]);
end
t2 = clock;
Time_1 = etime(t2,t1);
fprintf("Elapsed time using my own function: %f seconds.\n", Time_1);
%disp(result)

t1 = clock;
Time = 0;
for i = 1:m
        [result_2(i,1), result_2(i,2),every_time] = WithVLFeat(data(:, i), [data(:, 1:i-1) data(:, i+1:m)]);
	Time = Time + every_time;
end
t2 = clock;
Time_2 = etime(t2,t1);
%disp(result)
fprintf("Elapsed time including building kdtree is: %f seconds.\n", Time_2);
fprintf("Elapsed time excluding building kdtree is: %f seconds.\n", Time);

err_num = 0;
for i = 1:m
	if result(i,1)!=result_2(i,1)
		err_num = err_num + 1;
	end
end
err_rate = err_num / m	



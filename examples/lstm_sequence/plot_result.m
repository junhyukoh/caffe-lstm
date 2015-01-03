function [] = plot_result(file_name)

[fid, ~] = fopen(file_name);
A = fscanf(fid, '%f %f\n', [2 10000])';
fclose(fid);

plot(1:size(A, 1), A(:, 1), '.r', 'MarkerSize', 3);
hold on;
plot(1:size(A, 1), A(:, 2), '-k', 'LineWidth', 1);
hold off;

end
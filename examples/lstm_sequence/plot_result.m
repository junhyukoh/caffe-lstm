[fid, ~] = fopen('result.log');
A = fscanf(fid, '%f %f\n', [2 10000])';
fclose(fid);

plot(1:size(A, 1), A(:, 1), '.r');
hold on;
plot(1:size(A, 1), A(:, 2), '-b');
hold off;

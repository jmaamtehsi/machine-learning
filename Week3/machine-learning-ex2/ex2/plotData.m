function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

pos = X(y == 1, :);
neg = X(y == 0, :);

scatter(pos(:, 1), pos(:, 2), 'k+', 'LineWidth', 2)
scatter(neg(:, 1), neg(:, 2), 'o', 'LineWidth', 1, ...
    'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k')

% =========================================================================

hold off;

end

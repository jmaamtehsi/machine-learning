function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Note that the 1's are added to the X matrix in the input

h = X * theta;

% Make a new vector theta_j with the first element 0 to avoid 
% regularizing theta_0
theta_j = theta;
theta_j(1) = 0;

J = (1 / (2 * m)) * (h - y)' * (h - y) + ...
    (lambda / (2 * m)) * theta_j' * theta_j;

grad = (1 / m) * X' * (h - y) + (lambda / m) * theta_j;

% =========================================================================

grad = grad(:);

end

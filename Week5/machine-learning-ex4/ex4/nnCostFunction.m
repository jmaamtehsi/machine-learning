function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

% Theta1 = reshape(nn_params(1:(25 * 401)), 25, 401);

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Theta2 = reshape(nn_params(1+(25 * 401):end, 10, 26);

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1: Forward propagation

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Forward propagation
a2 = sigmoid(X * Theta1'); % 5000x401 matrix times 401x25 matrix -> 5000x25
a2 = [ones(size(a2, 1), 1) a2]; % 5000x25 -> 5000x26 matrix
a3 = sigmoid(a2 * Theta2'); % 5000x26 matrix times 26x10 matrix = 5000x10
h = a3;

eye_mat = eye(num_labels);
y_mat = eye_mat(y,:); % 5000x10 matrix of y values separated into 10 columns

% Remove first column from Theta1 and Theta2, which correspond to bias
% terms Theta_i0
reg_theta1 = Theta1(:,2:end);
reg_theta2 = Theta2(:,2:end);

J = (1 / m) * trace((-y_mat' * log(h) - (1 - y_mat)' * log(1 - h))) + ...
    (lambda / (2 * m)) * (sum(sum(reg_theta1.^2)) + sum(sum(reg_theta2.^2)));


% Part 2: Backpropagation

del1 = zeros(size(Theta1))
del2 = zeros(size(Theta2))

for i = 1:m                         % Dimensions
    % Forward prop
    a1 = X(i,:);                    % 1x401
    z2 = a1 * Theta1';              % 1x25 (=1x401 x 401x25)
    a2 = sigmoid(z2);               % 1x25
    a2 = [1 a2];                    % Add bias unit -> 1x26
    z3 = a2 * Theta2';              % 1x10 (=1x26 x 26x10)
    a3 = sigmoid(z3);               % 1x10
    y_i = y_mat(i,:);               % 1x10 <-- questionable line
    d3 = a3 - y_i;                  % 1x10
    
    % Backprop
    d2 = d3 * Theta2;              % 1x26 (=1x10 x 10x26)
    d2 = d2(:,2:end);               % 1x25 (remove the first term bias unit)
    d2 = d2 .* sigmoidGradient(z2);
    del2 = del2 + d3' * a2;         % 10x26 (=10x1 x 1x26)
    del1 = del1 + d2' * a1;         % 25x401 (25x1 x 1x401)
    
end

% Add column of zeros to beginning of reg_theta1 and reg_theta2 so that
% regularization does not apply to j = 0.
reg_theta1 = [zeros(size(reg_theta1, 1), 1) reg_theta1];
reg_theta2 = [zeros(size(reg_theta2, 1), 1) reg_theta2];

Theta1_grad = del1 ./ m + (lambda / m) * reg_theta1;
Theta2_grad = del2 ./ m + (lambda / m) * reg_theta2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

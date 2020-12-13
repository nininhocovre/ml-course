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

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

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

% forward propagation intermediate values
% Add ones to the X data matrix
alpha1 = [ones(m, 1) X];
z2 = alpha1 * Theta1';
%size(z2)
alpha2 = sigmoid(z2);
alpha2 = [ones(size(alpha2, 1), 1), alpha2];
z3 = alpha2 * Theta2';
%size(z3)
alpha3 = sigmoid(z3);


y_matrix = eye(num_labels)(y,:);

h = alpha3;

J = (1 / m) * sum(sum(-y_matrix .* log(h) - (1 - y_matrix) .* log(1 - h), 2));

J += (lambda / (2 * m)) * (sum(sum(power(Theta1(:,2:end), 2), 2)) + sum(sum(power(Theta2(:,2:end), 2), 2)));


%grad = (1 / m) * X' * (h - y)

%backpropagation
d2 = zeros(m, hidden_layer_size);
d3 = zeros(m, num_labels);

Delta1 = zeros(hidden_layer_size, (input_layer_size + 1));
Delta2 = zeros(num_labels, (hidden_layer_size + 1));

values = zeros(1, num_labels);
for val = 1:num_labels
  values(1,val) = val;
endfor

for t = 1:m
  d3(t, :) = alpha3(t, :) - (values == y(t));
  d2(t, :) = d3(t, :) * Theta2(:,2:end) .* sigmoidGradient(z2(t, :));
  
  Delta1 += d2' * alpha1;
  Delta2 += d3' * alpha2;
endfor

Theta1_grad = Delta1 ./ m;
Theta2_grad = Delta2 ./ m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

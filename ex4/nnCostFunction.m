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
%
%						Code Part1:cost function
%--------------------------------------------------------------------------------
%Hypothses = sigmoid([ones(m,1),( [ones(m,1) X]* Theta1')] * Theta2');
Y = eye(num_labels)(y,:);		%5000 X 10			
a1 = [ones(m, 1), X]; % results in [5000, 401]
a2 = sigmoid(a1 * Theta1'); % results in [25, 5000]
a2 = [ones(m,1) a2]; % results in [26, 5000]
h = sigmoid(a2 * Theta2'); % results in [10, 5000]
%costPositive = -Y .* log(h)';
%costNegative = (1 - Y) .* log(1 - h)';
%cost = costPositive - costNegative;
%J = (1/m) * sum(cost(:));

J = -1 / m * sum(sum(log(h) .* Y + log(1 - h) .* (1 - Y)));
%Part1.4 Regularized cost function
ThetaFilter1 = Theta1(:,2:end);
ThetaFilter2 = Theta2(:,2:end);

reg = lambda / (2 * m) *(sumsq(ThetaFilter1(:)) + sumsq(ThetaFilter2(:)));
J = J + reg;
 
%						Code Part2:
%--------------------------------------------------------------------------------
Delta1 = 0;
Delta2 = 0;
for i = 1:m
	a_1 = [1 X(i,:)];			%	1 X 401
	z_2 = a_1 * Theta1';		%	1 X 25
	a_2 = [1 sigmoid(z_2)];		%	1 X 26
	z_3 = a_2 * Theta2';		%	1 X 10
	a_3 = sigmoid(z_3);			%	1 X 10

	yi = Y(i, :);
	d_3 = a_3 - yi;				%	1 X 10 

	d_2 = d_3 * ThetaFilter2 .* sigmoidGradient(z_2);	%	1 X 25
	
	Delta2 = Delta2 + (d_3' * a_2);		%	10 X 26
	Delta1 = Delta1 + (d_2' * a_1);		%	25 X 401
end

Theta1_grad = 1 / m * (Delta1);
Theta2_grad = 1 / m * (Delta2);
%Part2.5: Regularized Neural Networks

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda / m * ThetaFilter1;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda / m * ThetaFilter2;


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
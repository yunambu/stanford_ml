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

% define Y, which is utilized for computing cost function J and consisted of m rows
% and K columns (i.e., #trainingExamples & #labels).
IdMatrix = eye(num_labels);
Y = zeros(m, num_labels);
for i = 1:m
    Y(i, :) = IdMatrix(y(i), :);
end

% calculate each activation unit in each hidden layer and finally obtain predictions.
X = a_ONE = [ones(m,1) X];
z_TWO = a_ONE * Theta1';
a_TWO = sigmoid(z_TWO);
a_TWO = [ones(size(z_TWO, 1), 1) a_TWO];
z_THREE = a_TWO * Theta2';
a_THREE = predictions = sigmoid(z_THREE);

% calculate cost function J and regularization function, which is utilized for giving
% penalty to cost function J
penalty = (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)));

J_noReg = (-1/m) * sum(sum(Y .* log(predictions) + (1 - Y) .* log(1 - predictions)));
J = J_noReg + penalty;

% -------------------------------------------------------------
% implement backpropagation to confirm the errors in previously calculated activation units
delta_THREE = a_THREE - Y;
delta_TWO = (delta_THREE * Theta2 .* sigmoidGradient([ones(size(z_TWO,1),1) z_TWO]))(:, 2:end);

% calculate the gradients both for input and hidden layers
Delta_ONE = delta_TWO' * a_ONE;
Delta_TWO = delta_THREE' * a_TWO;

% calculate the delivatives
Theta1_grad = (1/m) .* Delta_ONE + (lambda/m) * Theta1; 
Theta2_grad = (1/m) .* Delta_TWO + (lambda/m) * Theta2;

% -------------------------------------------------------------
% regularize the parameters that require to replace the first column in all rows with 0.
Reg_Theta1 = Theta1;
Reg_Theta2 = Theta2;
Reg_Theta1(:, 1) = 0;
Reg_Theta2(:, 1) = 0;

Reg_penalty = (lambda / (2 * m)) * (sum(sum(Reg_Theta1(:, 2:end).^2, 2)) + sum(sum(Reg_Theta2(:,2:end).^2, 2)));
J = J_noReg + Reg_penalty;

Theta1_grad = (1/m) .* Delta_ONE + (lambda/m) * Reg_Theta1;
Theta2_grad = (1/m) .* Delta_TWO + (lambda/m) * Reg_Theta2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

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
# predictions = zeros(m,num_labels); #initializing Htheta(x) variable

predictions = sigmoid(([ones(m,1) (sigmoid([ones(m,1) X] * Theta1'))]*Theta2')); # results in an M*num_labels matrix of predictions

newy = zeros(m,num_labels);

#Converting y from M*1 to M*num_labels
for i=1:m,
newy(i,y(i)) = 1;
end;

#Computing cost using newy (without regularization)

J = (((newy(:))'*log(predictions(:))) + ((1-newy(:))'*log(1-predictions(:))))/(-m);

#Computing regularization value
regularization = (lambda/(2*m))*((Theta1(:,[2 : size(Theta1,2)])(:)'*Theta1(:,[2 : size(Theta1,2)])(:))+(Theta2(:,[2 : size(Theta2,2)])(:)'*Theta2(:,[2 : size(Theta2,2)])(:)));#bias value not included

J = J + regularization;


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

#Implementation of the Back Propagation Algorithm - attempting without for_loop
newx = [ones(m,1) X];
a1 = newx; # Included a0-bias- for input layer (layer 1) ; Dimensions - M * 401

z2 = (a1 * Theta1'); # Dimensions - M * 25
a2 = ( [ones(m,1) sigmoid(z2)] ); # The Activation function of layer 2 ; Dimensions - M * 26 (Added a0-bias- for this layer)

z3 = (a2 * Theta2'); # Dimensions - M * 25
a3 = sigmoid(z3); # The Activation function of layer 3 ; Dimensions - M * 10

Sdelta3 = a3 - newy;#Error in a3 ; Dimensions - M * 10
Sdelta2 = (Sdelta3*Theta2 .* [ones(m,1) sigmoidGradient(z2)]);#Error in a2 ; Dimensions - M * (1 + 25) inclusive of bias unit error

Cdelta1 = Sdelta2(:,[2 : end])'*a1; #gradient of Theta1 -Sdelta2(0) has been excluded-
Cdelta2 = Sdelta3'*a2; #gradient of Theta2

Theta1_grad = Cdelta1/m;
Theta2_grad = Cdelta2/m;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

#Implementing regularized neural network gradient

Theta1_grad = Theta1_grad + [zeros(size(Theta1,1),1) (lambda/m)*Theta1(:,[2:end])];
Theta2_grad = Theta2_grad + [zeros(size(Theta2,1),1) (lambda/m)*Theta2(:,[2:end])];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Set the hypothesis h_X for logistic regression
h_X = sigmoid(X*theta);

% Set theta to a new value to skip the theta(1), not included in regularization
theta1 = [0 ; theta(2:size(theta),:) ];

% Set the regularization parameter with the new theta vector
p = lambda * (theta1' * theta1) / (2*m); 

% Set cost function with the regularization parameter
J = ( ( (-y)' * log(h_X) - (1-y)' * log(1 - h_X) ) / m ) + p;

% Set each gradient 
grad = (X' * (h_X - y) + lambda * theta1) / m;

% =============================================================

end

function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values

% m = number of training examples
m = length(y); 

thetaj = [0; theta(2:end)];

J = (1 / (2*m)) * (sum((X * theta - y) .^ 2) + lambda * (thetaj' * thetaj));
grad = (1/m) * (X' * (X * theta - y) + lambda * thetaj);

end
% =========================================================================



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


h_theta = sigmoid(X*theta);

v1 = -y .* log(h_theta);
v2 = (1-y) .* log(1 - h_theta);
J1 = 1 / m * sum(v1-v2);
J2 = lambda / (2 * m) * sum( theta([2:end]) .^ 2 );
J = J1 + J2;

theta_tmp = theta;
theta_tmp(1) = 0;
grad2 = lambda / m .* theta_tmp;

grad = 1/m .* sum((h_theta - y) .* X, 1) + grad2';

% =============================================================

end

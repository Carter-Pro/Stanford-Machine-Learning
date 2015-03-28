function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);
n = size(Xval, 1);
j = size(X, 2);
% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------
for i = 1:m
	theta_curve = ones(j + 1, 1);
	X_curve	= [ones(i,1) X(1:i, :)];
	y_curve	= y(1:i);
	Xval_curve = [ones(i,1) Xval(1:i, :)];
	yval_curve = yval(1:i);
	[theta_curve] = trainLinearReg(X_curve, y_curve, lambda);
	error_train(i) = 1 / (2 * i) * sumsq(X_curve * theta_curve - y_curve);
%	error_val(i) = 1 / (2 * i) * sumsq(Xval_curve * theta_curve - yval_curve);
%   !!!atention: when you compute error_val you must compute it over the entire cross validation set!!!
	error_val(i) = 1 / (2 * n) * sumsq([ones(n, 1) Xval] * theta_curve - yval);
end
% -------------------------------------------------------------

% =========================================================================

end

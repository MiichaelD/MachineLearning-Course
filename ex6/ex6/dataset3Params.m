function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;
% Best values found ^ with a prediction error of 0.03

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

values = [0.01 0.03 0.1 0.3 1 3 10 30];
lowest_error = 100000;

% fprintf('Looking for optimal C and sigma values\n');
index = 0;
for c = values
  for sig = values
    % fprintf('%d. - Trying C = %f and sigma = %f', ++index, c, sig);
    model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sig));
    error = mean(double(svmPredict(model, Xval) ~= yval));
    if (error <= lowest_error)
      C = c;
      sigma = sig;
      lowest_error = error;
      fprintf('Best C = %f, sigma = %f, error = %f\n', c, sig, error)
    end
  end
end
printf('Best C = %f, sigma = %f\n', C, sigma);

% =========================================================================

end

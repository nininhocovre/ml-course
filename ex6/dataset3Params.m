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
sigma = 0.3;

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

C_options = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_options = [0.01 0.03 0.1 0.3 1 3 10 30];
C_min = 0;
sigma_min = 0;
error_min = 0;

for C_val = C_options
  for sigma_val = sigma_options
    # train model with C and sigma values
    model= svmTrain(X, y, C_val, @(x1, x2) gaussianKernel(x1, x2, sigma_val));
    # run on Xval
    y_predict = svmPredict(model, Xval);

    error_calc = sum((yval - y_predict).^2);
    if (error_min == 0 || error_calc < error_min)
      error_min = error_calc;
      C_min = C_val;
      sigma_min = sigma_val;
    endif
  endfor
endfor

C = C_min;
sigma = sigma_min;

% =========================================================================

end

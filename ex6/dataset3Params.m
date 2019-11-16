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

fprintf('-------\n');
fprintf('start searching best [C, sigma] values\n');
error_min = inf;  % inf是内置函数，返回最大浮点数
values = [0.01 0.03 0.1 0.3 1 3 10 30];  % 硬编码测试序列，C sigma均从此数组取值，共8^2种组合

for _C = values
  for _sigma = values
    fprintf('Train and evaluate (on cross validation set) for\n[_C, _sigma] = [%f %f]\n', _C, _sigma);
    model = svmTrain(X, y, _C, @(x1, x2) gaussianKernel(x1, x2, _sigma));  % 使用匿名函数@(x1,x2)减少和固定kernel函数参数。
    e = mean(double(svmPredict(model, Xval) ~= yval));
    fprintf('prediction error: %f\n', e);
    if( e <= error_min )
      fprintf('error_min updated!\n');
      C = _C;
      sigma = _sigma;
      error_min = e;
      fprintf('[C, sigma] = [%f %f]\n', C, sigma);
    end
    fprintf('-------\n');
  end
end

fprintf('\nfinish searching.\nBest value [C, sigma] = [%f %f] with prediction error = %f\n\n', C, sigma, error_min);
fprintf('-------\n');


% =========================================================================

end

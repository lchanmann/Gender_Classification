function [W, a, e] = boosting_update( W, misclassified )
%BOOSTING compute classifier confidence and new weights in boosting
%
%   W - data point Weights
%   a - (alpha) classifier confidence
%   e - error
%

% error = sum of misclassified weights
e = sum( W(misclassified) );

% alpha
a = log( (1-e)/e ) / 2;

% update Weights for misclassified data-points
W(misclassified) = W(misclassified)/2/e;

% update Weights for correctly classified data-points
W(~misclassified) = W(~misclassified)/2/(1-e);
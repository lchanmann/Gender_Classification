function [ Hx ] = predict_Hx( model, X, t )
% PREDICT_HX predict boosting model
%
%   model - ensemble classifiers along with classifiers' strength
%       model.Classifiers - all classifiers in the ensemble
%       model.AlphaT      - strength of the classifiers
%

[N, ~] = size(X);
T = length(model.AlphaT);

% the number of classifiers used to do prediction
if nargin == 3
    T = min(T, t);
end

predicted = zeros(N, T);
for i=1:T
    if model.AlphaT(i) > 0
        predicted(:, i) = predict(model.Classifiers{i}, X);
    end
end
Hx = sign(predicted * model.AlphaT(1:T)');

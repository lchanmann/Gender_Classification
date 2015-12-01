function [ Hx ] = predict_Hx( model, X )
% PREDICT_HX predict boosting model
%
%   model - ensemble classifiers along with classifiers' strength
%       model.Classifiers - all classifiers in the ensemble
%       model.AlphaT      - strength of the classifiers
%

[N, ~] = size(X);
T = length(model.AlphaT);

predicted = zeros(N, T);
for i=1:T
    if model.AlphaT(i) > 0
        predicted(:, i) = predict(model.Classifiers{i}, X);
    end
end
Hx = sign(predicted * model.AlphaT');

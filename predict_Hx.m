function [ Hx ] = predict_Hx( models, alpha_t, X )
% PREDICT_HX predict boosting model

[N, ~] = size(X);
T = length(alpha_t);

predicted = zeros(N, T);
for i=1:T
    if alpha_t(i) > 0
        predicted(:, i) = predict(models{i}, X);
    end
end
Hx = sign(predicted * alpha_t');

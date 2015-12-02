function model = boosting( X_train, y_train, learner, T, varargin )
%BOOSTING perform classifier boosting for T iterations
%
%   X_train     - training predictors
%   y_train     - training labels
%   learner     - a weak learner to be boosted
%   T           - the number of iterations to boost
%

    % output model
    model.T = T;
    model.Classifiers = cell(1, T);
    model.AlphaT = zeros(1, T);

    tic;
    N = length(y_train);
    W = ones(N, 1) / N;
    % check for pre-defined Weights parameters
    for k=1:length(varargin)
        if strcmpi('Weights', varargin{k})
            W = varargin{k+1};
            break;
        end
    end

    % boosting
    for i=1:T
        fprintf('\tIteration %d: \n', i);
        classifier = learner(X_train, y_train ...
                , varargin{:} ...
                , 'Weights', W ...
            );
        model.Classifiers{i} = classifier;
        
        prediction = predict(classifier, X_train);
        misclassified = prediction ~= y_train;
        
        [W, a, e] = weights_update(W, misclassified);
        if e == 0
            break;
        end
        model.AlphaT(i) = a;
        fprintf('\t\t# of misclassified = %d out of %d\n', sum(misclassified), N);
        fprintf('\t\te = %0.5f\n', e);
        fprintf('\t\t');
        toc
        disp(' ');
    end
end

%% weights_update( W, misclassified )
function [W, a, e] = weights_update( W, misclassified )
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
end
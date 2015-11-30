clc;
display('_________________________________________________________');
display('                                                         ');
display('                    SVM Classifier                       ');
display('_________________________________________________________');
display(' ');
clear;
close all;

diary(['logs/svm_' num2str(datestr(now,'yyyymmdd.HHMM')) '.log']);
% load data
load('X.mat');
% use -1, +1 instead of 1, 2
y(y==1) = -1;
y(y==2) = +1;

% SVM training parameters
kernel = 'gaussian';
kernel_scale = 'auto';
% To use Quadratic Programming optimization (qp = 'L1QP')
optimization = 'SMO';
polynomial_order = 3;
% set upper-bound on \alpha. If C=Inf then svm don't allow
% mis-classification
C = 1;
% mis-classification cost
cost = [0 1; 1.5 0];
display('SVM parameters:');
fprintf('\tKernelFunction = %s\n', kernel);
if (strcmpi(kernel, 'polynomial'))
    fprintf('\tPolynomialOrder = %d\n', polynomial_order);
end
fprintf('\tKernelScale = %s\n', num2str(kernel_scale));
fprintf('\tSolver = %s\n', optimization);
fprintf('\tBoxConstraint = %0.2f\n', C);
fprintf('\tCost = [ %s ]\n', sprintf(' %0.1f ', cost));
disp(' ');

% data partition
k = 5;
CV = cvpartition(y, 'KFold', k);
accuracy = zeros(k, 1);

% boosting max iterations
T = 6;

models = cell(k, T);
alpha_t = zeros(k, T);
for j=1:k
    train_idx = CV.training(j);
    X_train = X(train_idx, :);
    y_train = y(train_idx, :);
    
    % boosting
    fprintf('Train boosted SVM for fold-%d...\n', j);
    tic;
    N = CV.TrainSize(j);
    W = ones(N, 1) / N;
    for i=1:T
        fprintf('\tIteration %d: \n', i);
        svm_m = fitcsvm(X_train, y_train ...
                , 'KernelFunction', kernel ...
                , 'KernelScale', kernel_scale ...
                ...% , 'ScoreTransform', 'sign' ...
                , 'Solver', optimization ...
                ...% , 'PolynomialOrder', polynomial_order ...
                , 'CacheSize', 'maximal' ...
                ...% , 'KKTTolerance', 0.1 ...
                , 'BoxConstraint', C ...
                ...% , 'CVPartition', CV ...
                ...% , 'OutlierFraction', 0.01 ...
                ...% , 'Verbose', 1, 'NumPrint', 1000 ...
                , 'Cost', cost ...,
                , 'Weight', W ...
            );
        prediction = predict(svm_m, X_train);
        models{j, i} = svm_m;
        
        [W, a, e] = boosting_update(W, prediction ~= y_train);
        if e == 0
            break;
        end
        alpha_t(j, i) = a;
        fprintf('\t\te = %0.5f\n', e);
        fprintf('\t\t');
        toc
        disp(' ');
    end
    
    % measure boosted svm performance on validation set
    test_idx = CV.test(j);
    X_test = X(test_idx, :);
    y_test = y(test_idx, :);
    predicted = zeros(length(y_test), T);
    
    for i=1:T
        if alpha_t(j, i) > 0
            predicted(:, i) = predict(models{j, i}, X_test);
        end
    end
    Hx = sign(predicted * alpha_t(j, :)');
    accuracy(j) = performance(Hx, y_test, 'Verbose');
end

% classification accuracy on training set
[N, ~] = size(X);
prediction = zeros(N, k);
for j=1:k
    prediction(:, j) = predict_Hx(models(j, :), alpha_t(j, :), X);
end
display('Training performance:');
performance(sign(prediction*ones(k,1)), y, 'Verbose');
disp(' ');

% K-Fold accuracy
fprintf('%d-Fold CV accuracy for boosted SVM = %0.5f', k, mean(accuracy));
display(accuracy);
diary off;
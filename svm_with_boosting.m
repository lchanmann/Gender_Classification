clc;
display('_________________________________________________________');
display('                                                         ');
display('                    SVM Classifier                       ');
display('_________________________________________________________');
display(' ');
clear;
close all;

diary(['logs/svm_' num2str(datestr(now,'yyyymmdd.HHMM')) '.log']);
% % load data
% load('X.mat');
% % use -1, +1 instead of 1, 2
% y(y==1) = -1;
% y(y==2) = +1;

% use the same data spliting
load('train-test_split.mat');

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
cost = [0 10; 29 0];
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

% training and test set partition
% [ X_trainset, y_trainset, X_testset, y_testset] = random_split(X, y, .1);

% 5-fold data partition
k = 5;
% CV = cvpartition(y_trainset, 'KFold', k);
accuracy = zeros(k, 1);

% boosting max iterations
T = 16;

models = cell(k, 1);
for j=1:k
    train_idx = CV.training(j);
    X_train = X_trainset(train_idx, :);
    y_train = y_trainset(train_idx, :);
    
    % boosting
    fprintf('Train boosted SVM for fold-%d...\n', j);
    models{j} = boosting(X_train, y_train, @fitcsvm, T ...
            , 'KernelFunction', kernel ...
            , 'KernelScale', kernel_scale ...
            ...% , 'ScoreTransform', 'sign' ...
            , 'Solver', optimization ...
            ...% , 'PolynomialOrder', polynomial_order ...
            , 'CacheSize', 'maximal' ...
            ...% , 'KKTTolerance', 0.1 ...
            , 'BoxConstraint', C ...
            ...% , 'OutlierFraction', 0.01 ...
            ...% , 'Verbose', 1, 'NumPrint', 1000 ...
            , 'Cost', cost ...,
        );    
    
    % measure boosted svm performance on validation set
    test_idx = CV.test(j);
    X_test = X_trainset(test_idx, :);
    y_test = y_trainset(test_idx, :);
    
    % ensemble prediction
    Hx = predict_Hx(models{j}, X_test);
    accuracy(j) = performance(Hx, y_test, 'Verbose');
end

% K-Fold accuracy
fprintf('%d-Fold CV accuracy for boosted SVM = %0.5f', k, mean(accuracy));
display(accuracy);

% classification accuracy on test set
[N, ~] = size(X_testset);
prediction = zeros(N, k);
for j=1:k
    prediction(:, j) = predict_Hx(models{j}, X_testset);
end
display('Performance on testset:');
performance(sign(prediction*ones(k,1)), y_testset, 'Verbose');
disp(' ');

diary off;
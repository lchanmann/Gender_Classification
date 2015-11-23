clc;
display('_________________________________________________________');
display('                                                         ');
display('                    SVM Classifier                       ');
display('_________________________________________________________');
display(' ');
clear;
close all;

% load data
load('X.mat');

% data partition
p = 0.4; % hold-out partition
CV = cvpartition(y, 'Holdout', p);
train_idx = training(CV);
test_idx = test(CV);

X_train = X(train_idx, :);
y_train = y(train_idx, :);
X_test = X(test_idx, :);
y_test = y(test_idx, :);

% SVM training
kernel = 'linear';
% To use Quadratic Programming optimization (qp = 'L1QP')
optimization = 'ISDA';
% set upper-bound on \alpha. If C=Inf then svm don't allow
% mis-classification
C = 1;
tic;
display('SVM training ...');
fprintf('\tKernelFunction = %s\n', kernel);
fprintf('\tSolver = %s\n', optimization);
fprintf('\tBoxConstraint = %0.2f\n', C);
fprintf('\t------------------------------\n');
svm = fitcsvm(X_train, y_train ...
        , 'KernelFunction', kernel ...
        ...% , 'ScoreTransform', 'sign' ...
        , 'Solver', optimization ...
        ...% , 'KKTTolerance', 0.1 ...
        , 'BoxConstraint', C ...
        , 'OutlierFraction', 0.01 ...
        ...% , 'Verbose', 1, 'NumPrint', 10000 ...
    );
fprintf('\t%d support vectors out of %d training samples!\n', ...
    sum(svm.IsSupportVector), length(y_train));
fprintf('\tWeights (see: svm.Beta), Bias (see: svm.Bias)\n');
disp(' ');
toc

% classification
predicted = predict(svm, X_test);
confusion = confusionmat(y_test, predicted);
accuracy = confusion([1 4]) / sum(confusion(:));

% print confusion and accuracy
display(confusion);
display('Gender accuracy =');
disp(' ');
fprintf('\t %0.4f ', accuracy);
disp(' ');
disp(' ');
display('Total accuracy =');
disp(' ');
fprintf('\t %0.4f ', sum(accuracy));
disp(' ');
disp(' ');

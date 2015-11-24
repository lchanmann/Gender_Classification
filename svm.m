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

% data partition
k = 5;
CV = cvpartition(y, 'KFold', k);

% SVM training
kernel = 'gaussian';
kernel_scale = 'auto';
% To use Quadratic Programming optimization (qp = 'L1QP')
optimization = 'SMO';
polynomial_order = 3;
% set upper-bound on \alpha. If C=Inf then svm don't allow
% mis-classification
C = 1;

tic;
display('SVM training ...');
fprintf('\tKernelFunction = %s\n', kernel);
if (strcmpi(kernel, 'polynomial'))
    fprintf('\tPolynomialOrder = %d\n', polynomial_order);
end
fprintf('\tKernelScale = %s\n', num2str(kernel_scale));
fprintf('\tSolver = %s\n', optimization);
fprintf('\tBoxConstraint = %0.2f\n', C);
svm = fitcsvm(X, y ...
        , 'KernelFunction', kernel ...
        , 'KernelScale', kernel_scale ...
        ...% , 'ScoreTransform', 'sign' ...
        , 'Solver', optimization ...
        ...% , 'PolynomialOrder', polynomial_order ...
        , 'CacheSize', 'maximal' ...
        ...% , 'KKTTolerance', 0.1 ...
        , 'BoxConstraint', C ...
        , 'CVPartition', CV ...
        ...% , 'OutlierFraction', 0.01 ...
        ...% , 'Verbose', 1, 'NumPrint', 1000 ...
    );
toc

% classification
N = sum(training(CV, 1));
for j = 1:svm.KFold
    model = svm.Trained{j};
    display(['Fold : ' num2str(j)]);
    fprintf('------------------------------\n');
    fprintf('\t%d support vectors out of %d training samples!\n', ...
        length(model.SupportVectors), N);
    disp(' ');
    
    test_idx = test(CV, j);
    X_test = X(test_idx, :);
    y_test = y(test_idx, :);
    svm_accuracy
end

display('KFold Prediction:');
predicted = kfoldPredict(svm);
confusion = confusionmat(y, predicted);
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
diary off;
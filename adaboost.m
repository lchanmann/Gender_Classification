clc;

display('_________________________________________________________');
display('                                                         ');
display('              SVM Classification Algorithm               ');
display('_________________________________________________________');
display(' ');
% load data
load('X.mat');

% train SVM with R samples from each class
R = 200;
X_1 = [X(1:R,:); X(N-R+1:N,:)];
y_1 = [y(1:R); y(N-R+1:N)];

tic
% Adaboost
model = fitensemble(X_1, y_1, 'AdaBoostM1', 200, 'Tree');

display('_________________________________________________________');
display('                                                         ');
display('        Classification Accuracy with Boosting            ');
display('_________________________________________________________');
display(' ');
% prediction accuracy
svm_accuracy = gender_accuracy(predict(model, X_1), y_1)
toc
pause;

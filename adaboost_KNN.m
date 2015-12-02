display('_________________________________________________________');
display('                                                         ');
display('                 KNN classifier Algorithm                ');
display('_________________________________________________________');
display(' ');
% load data
load('X.mat');

% train SVM with R samples from each class
N = length(y);
%R = 200;
X_1 = X;%[X(1:R,:); X(N-R+1:N,:)];
y_1 = y;%[y(1:R); y(N-R+1:N)];

tic
% Adaboost
template = classreg.learning.FitTemplate.make('KNN', 'NumNeighbors',100);
model = fitensemble(X_1, y_1, 'Subspace', 200, {template});


display('_________________________________________________________');
display('                                                         ');
display('          Classifier Accuracy with Boosting              ');
display('_________________________________________________________');
display(' ');
% prediction accuracy
boosted_accuracy = gender_accuracy(predict(model, X_1), y_1,1,2)
toc
pause;

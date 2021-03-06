display('_________________________________________________________');
display('                                                         ');
display('                Naive Base Classifier                    ');
display('_________________________________________________________');
display(' ');
diary(['logs/Bayes_' num2str(datestr(now,'yyyymmdd.HHMM')) '.log']);
%{
load('X.mat')
% use -1, +1 instead of 1, 2
y(y==1) = -1;
y(y==2) = +1;
[ train_X, train_Y, test_X, test_Y ] = random_split( X, y, .1 );
% 5-fold data partition
%CV = cvpartition(y_trainset, 'KFold', 5);
%}

load('train-test_split.mat');%use the same split for all experiments
k = CV.NumTestSets;

%% unboosted
display('_________________________________________________________');
display('                                                         ');
display('       Classifier Accuracy without Boosting              ');
display('_________________________________________________________');
display(' ');

model = fitcnb(X_trainset, y_trainset, 'Prior', 'uniform', 'CVPartition',CV);
Hx_model.Classifiers = model.Trained;
Hx_model.AlphaT = ones(1,k);
performance(predict_Hx(Hx_model, X_testset), y_testset, 'Verbose');

%% Boosting
display('_________________________________________________________');
display('                                                         ');
display('          Classifier Accuracy with Boosting              ');
display('_________________________________________________________');
display(' ');

% 5-fold data partition
%CV = cvpartition(y_trainset, 'KFold', 5);
%Use same split for all experiments!
k = CV.NumTestSets;

% boosting max iterations
T = 16;

alpha_t = zeros(k, T);
boosted_models = cell(k, 1);
boosted_accuracy = zeros(k,1);
misclassifieds = zeros(k,T);
h = waitbar(0,'Crossvalidating boosted Naive Bayes');
for j=1:k
    train_idx = CV.training(j);
    X_train = X_trainset(train_idx, :);
    y_train = y_trainset(train_idx, :);
    
    % boosting
    waitbar((2*j-1)/(2*k),h,sprintf('Boosting Naive Bayes fold %d of %d',neighbors,j,k));
    fprintf('Train boosted Naive Bayes for fold-%d...\n', j);
    [boosted_models{j}, misclassifieds(j,:)] = boosting(X_train, y_train, @fitcnb, T, ...
                                 'Prior','uniform');    
    
    waitbar((2*j)/(2*k),h,sprintf('Crossvalidating boosted Naive Bayes fold %d of %d',j,k));
    % measure boosted svm performance on validation set
    test_idx = CV.test(j);
    X_test = X_trainset(test_idx, :);
    y_test = y_trainset(test_idx, :);
    
    % ensemble prediction
    Hx = predict_Hx(boosted_models{j}, X_test);
    boosted_accuracy(j) = performance(Hx, y_test, 'Verbose');
end
delete(h)

% K-Fold accuracy
fprintf('%d-Fold CV accuracy for boosted Naive Bayes = %0.5f', k, mean(boosted_accuracy));
display(boosted_accuracy);

figure
plot(misclassifieds')
set(gca,'Ydir','reverse')
title('Number of training set misclassifications by boosted learners (Naive Bayes)')
legend('1st fold','2nd fold','3rd fold','4th fold','5th fold')
xlabel('Iteration of Adaboost')

% classification accuracy on test set
[N, ~] = size(X_testset);
prediction = zeros(N, k);
for j=1:k
    prediction(:, j) = predict_Hx(boosted_models{j}, X_testset);
end

%%
display('Performance on testset:');
accTest = zeros(1, T);
for t=1:T
    prediction = zeros(N, k);
    for j=1:k
        prediction(:, j) = predict_Hx(boosted_models{j}, X_testset, t);
    end
    fprintf('t = %d', t);
    votes=prediction*ones(k,1); %sums each fold's models' votes
    accTest(t) = performance(sign(votes), y_testset, 'Verbose');
    disp(' ');
end
plot(accTest)
title('Accuracy of Boosted Naive Bayes classifiers')
xlabel('Number of Boosting Rounds in classifier')

%%
diary off
%Finds the optimal number of neighbors for KNN
display('_________________________________________________________');
display('                                                         ');
display('                 KNN classifier Algorithm                ');
display('_________________________________________________________');
display(' ');

diary(['logs/KNN_' num2str(datestr(now,'yyyymmdd.HHMM')) '.log']);
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


%% Evaluate unboosted KNN

display('_________________________________________________________');
display('                                                         ');
display('       Classifier Accuracy without Boosting              ');
display('_________________________________________________________');
display(' ');

%maximum number of neighbors to test
n = 100;%Alternative ns: 530; %Number of images of Bush (most common face); sum(y==2); %size of smallest class

male_accuracy = zeros(n, 1);
female_accuracy = zeros(n, 1);
accuracy = zeros(n, 1);


tic
%530 neighbors takes aproximately 15 minutes
h = waitbar(0,'Searching for optimal number of neighbors...');
for i=1:n
    model = fitcknn(X_trainset, y_trainset, 'NumNeighbors', i, 'Prior', 'uniform', 'CVPartition',CV);
    waitbar((i*(i+1)/2)/(n*(n+1)/2), h, sprintf('Testing optimality for %d of %d neighbors',i,n));
    Hx_model.AlphaT = ones(1,k);
    Hx_model.Classifiers = model.Trained;
    
    %{
    noboost_accuracy = gender_accuracy(predict_Hx(Hx_model, X_testset), y_testset,-1,1);
    male_accuracy(i) = noboost_accuracy.males;
    female_accuracy(i) = noboost_accuracy.females;
    accuracy(i) = noboost_accuracy.global;
    %}
    [accuracy(i), confusion] = performance(predict_Hx(Hx_model, X_testset),y_testset);
    male_accuracy(i) = confusion(1,1)/sum(y_testset == -1);
    female_accuracy(i) = confusion(2,2)/sum(y_testset == 1);
end
delete(h)
t=toc;
fprintf('Search finished after %d h, %d min, %f sec\n',floor(t/60^2),floor(t/60),rem(t,60));

figure
hold on
plot(male_accuracy)
plot(female_accuracy)
plot(accuracy)
legend('male accuracy','female accuracy','global accuracy')
xlabel('number of neighbors')
title('Accuracy results for K-nearest neighbors')

[best_male_accuracy, neighbors] = max(male_accuracy)
[best_female_accuracy, neighbors] = max(female_accuracy)
[best_global_accuracy, neighbors] = max(accuracy)

%% Evaluate boosted KNN
%yes, this is the same code over again. Following good programming practice
%in this case would require making the caller construct an anonymous
%function to use the function, which is more trouble than it's worth

display('_________________________________________________________');
display('                                                         ');
display('          Classifier Accuracy with Boosting              ');
display('_________________________________________________________');
display(' ');

%{
%Currently using neighbors as best from unboosted KNN
%Alternative: Find best for each fold and boost with that
noboost.accuracy = zeros(k, n);
noboost.accuracy_males = zeros(k, n);
noboost.accuracy_females = zeros(k, n);
unboosted_models = cell(k, 1);
%}

% boosting max iterations
T = 16;

alpha_t = zeros(k, T);
boosted_models = cell(k, 1);
boosted_accuracy = zeros(k,1);
misclassifieds = zeros(k,T);
h = waitbar(0,sprintf('Crossvalidating boosted %d-NN',neighbors));
for j=1:k
    train_idx = CV.training(j);
    X_train = X_trainset(train_idx, :);
    y_train = y_trainset(train_idx, :);
    
    % boosting
    waitbar((2*j-1)/(2*k),h,sprintf('Boosting %d-NN fold %d of %d',neighbors,j,k));
    fprintf('Train boosted %d-NN for fold-%d...\n', neighbors, j);
    [boosted_models{j}, misclassifieds(j,:)] = ...
                            boosting(X_train, y_train, @fitcknn, T, ...
                            'NumNeighbors', neighbors,'Prior','uniform');    
    
    waitbar((2*j)/(2*k),h,sprintf('Crossvalidating boosted %d-NN fold %d of %d',neighbors,j,k));
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
fprintf('%d-Fold CV accuracy for boosted %d-NN = %0.5f', k, neighbors, mean(boosted_accuracy));
display(boosted_accuracy);

figure
surf(misclassifieds)
title(sprintf('Number of training set misclassifications by boosted learners (%d-NN)',neighbors))
set(gca,'Zdir','reverse')
ylabel('Fold number')
xlabel('Iteration of Adaboost')
set(gca,'Ydir','reverse')
set(gca,'Xdir','reverse')



% classification accuracy on test set
[N, ~] = size(X_testset);
prediction = zeros(N, k);
for j=1:k
    prediction(:, j) = predict_Hx(boosted_models{j}, X_testset);
end
display('Performance on testset:');
votes=prediction*ones(k,1); %sums each fold's models' votes
performance(sign(votes), y_testset, 'Verbose');
disp(' ');

%% End
diary off;
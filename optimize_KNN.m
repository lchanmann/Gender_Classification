%Finds the optimal number of neighbors for KNN
display('_________________________________________________________');
display('                                                         ');
display('                 KNN classifier Algorithm                ');
display('_________________________________________________________');
display(' ');
load('X.mat')
% use -1, +1 instead of 1, 2
y(y==1) = -1;
y(y==2) = +1;

[ train_X, train_Y, test_X, test_Y ] = random_split( X, y, .1 );

%% Evaluate unboosted KNN

display('_________________________________________________________');
display('                                                         ');
display('       Classifier Accuracy without Boosting              ');
display('_________________________________________________________');
display(' ');

%maximum numberof neighbors to test
n = 530; %Number of images of Bush (most common face)
%n = sum(y==2); %size of smallest class

male_accuracy = zeros(n, 1);
female_accuracy = zeros(n, 1);
accuracy = zeros(n, 1);


tic
%530 neighbors takes aproximately 15 minutes
h = waitbar(0,'Searching for optimal number of neighbors...');
for i=1:n
    model = fitcknn(train_X, train_Y, 'NumNeighbors', i, 'Prior', 'uniform');
    waitbar((i*(i+1)/2)/(n*(n+1)/2), h, sprintf('Testing optimality for %d of %d neighbors',i,n));
    noboost_accuracy = gender_accuracy(predict(model, test_X), test_Y,-1,1);
    male_accuracy(i) = noboost_accuracy.males;
    female_accuracy(i) = noboost_accuracy.females;
    accuracy(i) = noboost_accuracy.global;
end
close(h)
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

% 5-fold data partition
k=1;%k = 5;
%CV = cvpartition(y_trainset, 'KFold', k);
accuracy = zeros(k, 1);

% boosting max iterations
T = 6;

models = cell(k, 1);
% alpha_t = zeros(k, T);
j=1;%for j=1:k
    %train_idx = CV.training(j);
    X_train = train_X;%X(train_idx, :);
    y_train = train_Y;%(train_idx, :);
    
    % boosting
    fprintf('Train boosted %d-NN for fold-%d...\n', neighbors, j);
    models{j} = boosting(X_train, y_train, @fitcknn, T, ...
                         'NumNeighbors',neighbors,'Prior','uniform');    
    
    % measure boosted svm performance on validation set
    %test_idx = CV.test(j);
    X_test = test_X;%X(test_idx, :);
    y_test = test_Y;%y(test_idx, :);
    
    % ensemble prediction
    Hx = predict_Hx(models{j}, X_test);
    accuracy(j) = performance(Hx, y_test, 'Verbose');
%end

% K-Fold accuracy
fprintf('%d-Fold CV accuracy for boosted %d-NN = %0.5f', neighbors, k, mean(accuracy));
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

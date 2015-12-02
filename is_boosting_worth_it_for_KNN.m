display('_________________________________________________________');
display('                                                         ');
display('              KNN Classifier Algorithm                   ');
display('_________________________________________________________');
display(' ');
load('X.mat')

cols = 90;
neighbors = 100;

males = sum(y==1);
females = sum(y==2);

trainsize = max(round(males*.4),females);

            %use only when trainsize is small enough to leave some females for testing!
test_X = X; %X(trainsize:(length(y)-trainsize),:);
test_Y = y; %Y(trainsize:(length(y)-trainsize));

%Construct bayesian networks to ESTIMATE the accuracy
%of the proper implementation (cross-validated, etc)
noboost_BN_est = train_model(X, y, @fitcknn, cols, trainsize, 'NumNeighbors', neighbors)
boosted_BN_est = train_model(X, y, @fitcknn, cols, min(males,females), 'NumNeighbors', neighbors)
display('_________________________________________________________');
display('                                                         ');
display('         Classifier Accuracy without Boosting            ');
display('_________________________________________________________');
display(' ');
noboost_accuracy_estimate = gender_accuracy(predict(noboost_BN_est, test_X), test_Y,1,2)
display('_________________________________________________________');
pause;
display('                                                         ');
display('           Classifier Accuracy with Boosting             ');
display('_________________________________________________________');
display(' ');
boosted_accuracy_estimate = gender_accuracy(predict(boosted_BN_est, test_X), test_Y,1,2)
pause;

if(noboost_accuracy_estimate.males < noboost_accuracy_estimate.females)
    if(noboost_accuracy_estimate.males < boosted_accuracy_estimate.males)
        boosting_improvement_estimate = noboost_accuracy_estimate.females/noboost_accuracy_estimate.males
    else
        disp('boosting is not worth it');
        return
    end
else
    if(noboost_accuracy_estimate.females < boosted_accuracy_estimate.females)
        boosting_improvement_estimate = noboost_accuracy_estimate.males/noboost_accuracy_estimate.females
    else
        disp('boosting is not worth it');
        return
    end
end
if boosting_improvement_estimate > 1
    disp('boosting is worth it. Get on to it!');
else
    disp('boosting is not worth it. Extract more features');
end

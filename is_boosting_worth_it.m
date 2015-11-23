display('_________________________________________________________');
display('                                                         ');
display('                 Nive Base Classifier                    ');
display('_________________________________________________________');
display(' ');
load('X.mat')

cols = 90;

males = sum(y==1);
females = sum(y==2);

%Construct bayesian networks to ESTIMATE the accuracy
%of the proper implementation (cross-validated, etc)

noboost_BN_est = train_model(X, y, @fitNaiveBayes);
boosted_BN_est = train_model(X, y, @fitNaiveBayes, cols, min(males,females));
display('_________________________________________________________');
display('                                                         ');
display('       Classifier Accuracy without Boosting              ');
display('_________________________________________________________');
display(' ');
noboost_accuracy_estimate = gender_accuracy(predict(noboost_BN_est, X), y)
pause;
display('_________________________________________________________');
display('                                                         ');
display('          Classifier Accuracy with Boosting              ');
display('_________________________________________________________');
display(' ');
boosted_accuracy_estimate = gender_accuracy(predict(boosted_BN_est, X), y)
pause;
if(noboost_accuracy_estimate.males < noboost_accuracy_estimate.females)
    if(noboost_accuracy_estimate.males < boosted_accuracy_estimate.males)
        boosting_improvement_estimate = noboost_accuracy_estimate.females/noboost_accuracy_estimate.males
    else
        disp('boosting is not worth it');
    end
else
    if(noboost_accuracy_estimate.females < boosted_accuracy_estimate.females)
        boosting_improvement_estimate = noboost_accuracy_estimate.males/noboost_accuracy_estimate.females
    else
        disp('boosting is not worth it');
    end
end
if boosting_improvement_estimate > 1
    disp('boosting is worth it. Get on to it!');
else
    disp('boosting is not worth it. Extract more features');
end

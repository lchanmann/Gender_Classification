clc;
c=0;
while (c~=3)
    clc;
    display('_______________________________________________');
    display('                                               ');
    display('        Classifier without Boosting            ');
    display('_______________________________________________');
    display(' ');
    display('     1: SVM Algorithm / with Boosting          ');
    display('     2: KNN Algorithm / with Boosting          ');
    display('     3: Nive Base Algorithm / with Boosting    ');
    display('     4: Go back                        ');
    display('_______________________________________________');
    c1=input('Select the classifier No. : ');
        if c1==1
            % using SVM wihout Boosting
            svm;
            % using SVM with Boosting
            adaboost;
        elseif c1==2
            is_boosting_worth_it_for_KNN;
            %adaboost_KNN;
        elseif c1==3
            is_boosting_worth_it;
        else
            %display('You choise a wrong number, please try again ');
            c1=4;
            break;
        end
end   

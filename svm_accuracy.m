predicted = predict(model, X_test);
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
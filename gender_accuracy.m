function accuracy = gender_accuracy(predicted_labels, actual_labels)
%Returns an accuracy struct with the fields "global accuracy", "male
%accuracy" and "female accuracy" indicating, respectively what ratio of the
%males was correctly classified, what ratio of the females was correctly
%classified, and what ratio was correctly classified overall. Assumes
%males=1 and females=2.

    accuracy.global = sum(predicted_labels == actual_labels)/length(actual_labels);

    males = predicted_labels(actual_labels==1);
    accuracy.males = sum(males==1)/length(males);

    females = predicted_labels(actual_labels==2);
    accuracy.females = sum(females==2)/length(females);
end
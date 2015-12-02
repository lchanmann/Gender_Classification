function accuracy = gender_accuracy(predicted_labels, actual_labels,male_label,female_label)
%Returns an accuracy struct with the fields "global accuracy", "male
%accuracy" and "female accuracy" indicating, respectively what ratio of the
%males was correctly classified, what ratio of the females was correctly
%classified, and what ratio was correctly classified overall. 

    accuracy.global = sum(predicted_labels == actual_labels)/length(actual_labels);

    males = predicted_labels(actual_labels==male_label);
    accuracy.males = sum(males==male_label)/length(males);

    females = predicted_labels(actual_labels==female_label);
    accuracy.females = sum(females==female_label)/length(females);
end
function [ acc, confusion ] = performance( prediction, actual, varargin )
%PERFORMANCE evaluate the performance of prediction 
%   against true labels

confusion = confusionmat(actual, prediction);
acc = sum(confusion([1 4])) / sum(confusion(:));

if ~isempty(varargin) && sum(strcmpi('verbose', varargin(:))) > 0
    % print confusion and accuracy
    display(confusion);
    display('Aaccuracy =');
    disp(' ');
    fprintf('\t %0.4f ', acc);
    disp(' ');
    disp(' ');
end
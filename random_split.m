function [ train_X, train_Y, test_X, test_Y ] = random_split( X, Y, r )
%Splits X (data) and Y (labels) of a dataset containing classes 1 and 2
%into random training and testing sets, with the testing set
%containing the same number of elements of both classes. 'r' is used to
%specify what ratio of the smallest class should be included in the testing
%set.
% Inputs:
%   X - The dataset to be classified, each row must represent a datapoint
%   Y - The labels of the dataset. (linear array) The labels are assumed to 
%       be 1 and 2 and the dataset is assumed to be sorted by labels
%   r - A number between 0 and 1. Indicates how much of the smallest
%       class will be put in the testing set. The actual ammount return may
%       be slightly less because no repeated data points are used.
    
    assert(any(size(Y)==1)) %Labels must be a linear array
    assert(size(X,1)==length(Y)) %There must be exactly one label for each data point
    
    N=length(Y);
    [M,C]=min([sum(Y==-1),sum(Y==+1)]);
    
    d = floor(r*M); %number of random values to generate
    
    if(C==2)
        test_indices = [rand(d,1)*(N-M); (N-M+1)+rand(d,1)*M];
    else
        test_indices = [rand(d,1)*M; (M+1)+rand(d,1)*(N-M)];
    end
    
    test_indices = unique(round(test_indices));
    
    test_X = X(test_indices, :);
    test_Y = Y(test_indices);
    
    train_X = X(setdiff(1:N,test_indices), :);
    train_Y = Y(setdiff(1:N,test_indices));
end


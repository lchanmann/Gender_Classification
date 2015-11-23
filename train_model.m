function M = train_model( X, y, fxn, f, s, varargin)
%Constructs the specified classifier on a subset of the data. The
%classifier should be compatible with PREDICT
% Inputs:
%   X - The data to classify, ech row being a data sample
%   y - The data labels corresponding to the rows of X
%   fxn - classifier constructor function
%   f - The number of features to take (optional. Only the first f columns 
%       of X be taken. If not provided, all columns will be used
%   s - The number of samples to take (optional). The first s rows 
%       and last s rows will be taken. If not provided, all rows will be
%       used.
%   ... - all other parameters are passed on to fxn
% Outputs:
%   BN - A matlab classifier object. Can be called with predict(BN,X)
%        to find what labels it produces

    assert(any(size(y)==1)) %verify class vector is linear
    assert(size(X,1)==length(y)) %verify there's a label for each data point
    
    %parse arguments
    if nargin>=4
        features = 1:f;
        
        if nargin>=5
            samples = [1:s, (length(y)-s+1) : length(y)];
            X=X(samples, features);
            y=y(samples);
        else
            X=X(:,features);
        end
    end
    
    M = fxn(X, y, varargin{:});
end


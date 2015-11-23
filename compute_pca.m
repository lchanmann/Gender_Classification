%% IDoing PCA on the LFW dataset...
 display('_________________________________________________________');
 display('                                                         ');
 display('              Feature Extraction using PCA               ');
 display('_________________________________________________________');
 display(' ');
 clear ; close all; clc
%% Load Face dataset...
    load ('LFW.mat');
    % Display the first 100 faces in the dataset
    displayData(X(1:25, :));
%% visualize the eigenvectors...
    % Run PCA and visualize the eigenvectors which are in this case eigenfaces
    % We display the first 36 eigenfaces.
     fprintf(['\n Running PCA on LFW face dataset.\n' ...
              '(This mght take a few minute ...)\n\n']);
    % Before running PCA, it is important to first normalize X by subtracting 
    % the mean value from each feature...
    [X_norm, mu, sigma] = featureNormalize(X);
    temp = find(y == 2);
    numel(temp);
    numb = max(temp) - min(temp);
    glue = [(randperm(numb))'; temp];
%% Run PCA ...
    [U, S] = pca(X_norm(glue,:));
    pause;
    % Visualize the top 36 eigenvectors found
    displayData(U(:, 1:36)');
    fprintf('Program paused. Press enter to continue.\n');
    pause;
%% Dimension Reduction for Faces
    %  Project images to the eigen space using the top k eigenvectors 
    %  using a machine learning algorithm 
    fprintf('\nDimension reduction for face dataset.\n\n');
    
    % select dimension reduction space
    K = 900;
    
    % face images projection
    Z = projectData(X_norm, U, K);
    fprintf('The projected data Z has a size of: ')
    fprintf('%d ', size(Z));
    fprintf('\n\nProgram paused. Press enter to continue.\n');
    pause;

%% Visualization of Faces after PCA Dimension Reduction
    % Project images to the eigen space using the top K eigen vectors and 
    % visualize only using those K dimensions
    % Compare to the original input, which is also displayed
    fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');
    X_rec  = recoverData(Z, U, K);
    
    % Display normalized data
    subplot(1, 2, 1);
    displayData(X_norm(1:25,:));
    title('Original faces');
    axis square;
    
    % Display reconstructed data from only k eigenfaces
    subplot(1, 2, 2);
    displayData(X_rec(1:25,:));
    title('Recovered faces');
    axis square;
    fprintf('Program paused. Press enter to continue.\n');
    pause;
    X = Z;
    %X.mat has the information about the dataset mu,sigma
     save ('X_norm.mat', 'mu', 'sigma', 'U', 'K');
    % Final reduction dataset
     save ('X.mat', 'X', 'y');
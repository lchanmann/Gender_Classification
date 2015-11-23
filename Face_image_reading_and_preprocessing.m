%% Initialization...
 clear ; close all; clc
% set face image dimension
 input_dimension_size  = 6400;  % 80x80 Input Images of Digits
% set number of labels
 num_labels = 2;        
% Load Training Data
 display('_________________________________________________________');
 display('                                                         ');
 display('               LFW faces image reading                   ');
 display('_________________________________________________________');
 display(' ');
%% Male face images reading...
% set the direction
fprintf(['\n LFW faces image reading and doing the preprocessing ... \n' ...
         '(This mght take a few minute ...)\n\n']);
% set direction for reading 
cd ('LFW_Cropping\male\');
D = dir('*.jpg');
X = zeros(numel(D), input_dimension_size);
fprintf('Images which are in processing...\n');
fprintf('_________________________________\n');
    for i = 1:numel(D)
         fprintf('image no. %d\n',i');
         temp = (imread(D(i).name));
         % convert images to gray scale           
         temp = rgb2gray(temp);
         % do Histogram equlization
         temp= histeq(temp);
         % image resizing to 80x80
         temp = imresize(temp,[80,80]);
         X(i,:) = (temp(:))';
    end;
y = ones (numel(D), 1);
display('Done... \n');
clear temp;
%% Female face images reading...
% set the direction
clc;
cd('../..');
cd ('LFW_Cropping\female\');
D2 = dir('*.jpg');
X2 = zeros(numel(D2), input_dimension_size);
fprintf('Images which are in processing...\n');
fprintf('_________________________________\n');
    for i = 1:numel(D2)
         fprintf('image no. %d\n',i');
         temp = (imread(D2(i).name));
         % conver each image to grayscale
         temp = rgb2gray(temp);
         % do Histogram equlization
         temp= histeq(temp);
         % image resizing to 80x80
         temp = imresize(temp,[80,80]);
         X2(i,:) = (temp(:))';
    end;
X = [X; X2];
%X = X / 256;
y2 = ones(numel(D2), 1) * 2;
y = [y; y2];
% datset pqrameters
clear X2, clear y2;
display('Done... \n');
cd('.../.../../..');
%% Save the mat file dataset
  save ('LFW.mat', 'X', 'y'); %X is the face images and Y is the label
  display('\LFW_face_Detected processing...(Done) ...');

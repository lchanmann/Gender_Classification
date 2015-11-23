function [] = dataset_plot()
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
load adil.mat;
plot(adil(1:200,1),adil(1:200,2),'rx');
hold on;
plot(adil(201:400,1),adil(201:400,2),'b.');
end
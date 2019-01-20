clear ; close all; clc; addpath('functions')
%% =========== Regularized Linear Regression and Bias-Variance ========= %%
%  written by : JAWAD FAYAZ (email: jfayaz@uci.edu)

%  ------------- Instructions -------------- %
%  INPUT:
%  Input Variables must be in the form of .mat file and must be in same directory
%  Input Variables should include:
%   "Exdata"  --> this should include following variables:
%      'X'      -->  (m,1) vector containing Train data 
%      'y'      -->  (m,1) vector containing Train data
%      'Xval'   -->  (n,1) vector containing Cross-Validation data
%      'yval'   -->  (n,1) vector containing Cross-Validation data
%      'Xtest'  -->  (n,1) vector containing Test data
%      'ytest'  -->  (n,1) vector containing Test data
%
%  OUTPUT:
%  Output will be provided in following variables:  
%  "theta"        --> Vector (2,1) containing the Linear Regression coeffecients: Intercept and the Slope  
%  "grad"         --> Vector (2,1) contains gradients at theta
%  "J"            --> (1,1) Cost Function value
%  "lambda"       --> (1,1) Regularized regression parameter

%%%%% ============================================================= %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ====================== USER INPUTS =============================== %%
%%% Provide your .mat file name here  
Matlab_Data_Filename = 'Exdata.mat';


%%%%%%================= END OF USER INPUT ========================%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ---------- Plotting Data ----------
load (Matlab_Data_Filename);

% m = Number of examples
m = size(X, 1);

% Plot training data
figure(1)
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('X','fontWeight','bold')
ylabel('y','fontWeight','bold')


%% ---------- Regularized Linear Regression Cost ----------
theta = [1 ; 1];
J = linearRegCostFunction([ones(m, 1) X], y, theta, 1);

fprintf(['Cost at theta = [1 ; 1]: %f \n'], J);


%% ---------- Regularized Linear Regression Gradient ----------
theta = [1 ; 1];
[J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);

fprintf(['Gradient at theta = [1 ; 1]:  [%f; %f] \n'],grad(1), grad(2));


%% ---------- Train Linear Regression ----------
%  trainLinearReg function will use the cost function to train regularized linear regression.

%  Train linear regression with lambda = 0
lambda = 0;
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);

%  Plot fit over the data
hold on;
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
legend('Data Points', 'Regularized Linear Regression')
hold off;


%% ---------- Learning Curve for Linear Regression ----------
lambda = 0;
[error_train, error_val] = learningCurve([ones(m, 1) X], y,[ones(size(Xval, 1), 1) Xval], yval, lambda);

figure(2)
plot(1:m, error_train, 1:m, error_val, 'LineWidth', 1.5);
title('Learning Curve for Linear Regression')
legend('Train', 'Cross Validation')
xlabel('Number of Training Examples','fontWeight','bold')
ylabel('Error','fontWeight','bold')

fprintf('# Training_Examples\tTrain_Error\tCross-Validation_Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

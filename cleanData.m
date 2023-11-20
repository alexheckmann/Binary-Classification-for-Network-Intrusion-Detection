function [train_df, test_df, numericalPredictors, categoricalPredictors] = cleanData(train_df, test_df, removedFeatures)
% cleanData
%
% Removes specified features from the train and test data, and separates the numerical and categorical predictors.
%
% Syntax:
%   [train_df, test_df, numericalPredictors, categoricalPredictors] = cleanData(train_df, test_df, removedFeatures)
%
% Inputs:
%   train_df:            Train data.
%   test_df:             Test data.
%   removedFeatures:     A cell array of feature names to be removed from the train and test data.
%
% Outputs:
%   train_df:            Processed train data with specified features removed.
%   test_df:             Processed test data with specified features removed.
%   numericalPredictors: Indices of numerical predictors in the processed data.
%   categoricalPredictors: Indices of categorical predictors in the processed data.
try
    train_df(:, removedFeatures) = [];
    test_df(:, removedFeatures) = [];
    disp("Removed unnecessary features...");
    featureAmount = size(train_df, 2) - 1;
    disp("Amount of features remaining: " + featureAmount);

    % subtracting one for the label
    nPredictors = size(train_df, 2) - 1;
    numericalPredictors = [1, 5:nPredictors];
    categoricalPredictors = 2:4;

catch
    warning("Listed features are not found in the dataset.")
end
end
function [correlationMatrix] = getCorrelation(dataset, numericalFeatures)
% getCorrelation - Computes the correlation matrix for a given dataset and list of numerical features.
%
% INPUTS:
%   dataset - a table containing the data
%   numericalFeatures - a list of indices or names of the numerical features in the dataset
%
% OUTPUTS:
%   correlationMatrix - a matrix containing the correlations between all pairs of numerical features

% Extract the numerical features from the dataset
corrFeatures = dataset(:, numericalFeatures);

% Convert the table to a matrix by extracting its data using curly brackets
corrFeatures = corrFeatures{:,:};

% Calculate the correlation matrix using the corr function
correlationMatrix = corr(corrFeatures);

end

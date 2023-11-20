function [heatMap] = corrheatmap(correlationMatrix, xvalues, yvalues)
    % corrheatmap
    %
    % Plots a heatmap of a correlation matrix.
    % Source: https://uk.mathworks.com/matlabcentral/answers/1750730-how-to-plot-correlation-coefficient-matrix-plot
    %
    % Syntax:
    %   [heatMap] = corrheatmap(correlationMatrix, xvalues, yvalues)
    %
    % Inputs:
    %   correlationMatrix:  A matrix of correlations.
    %   xvalues:           Labels for the x-axis of the heatmap.
    %   yvalues:           Labels for the y-axis of the heatmap.
    %
    % Outputs:
    %   heatMap:           A heatmap object.

    % finding the values in upper triangle of the correlation matrix
    isUpper = logical(triu(ones(size(correlationMatrix)),1));
    % removing the upper triangle of corr matrix to simplify interpretation
    correlationMatrix(isUpper) = NaN;
    heatMap = heatmap(xvalues, yvalues, correlationMatrix, "MissingDataColor","w");

end

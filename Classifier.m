classdef Classifier
    % Classifier
    %
    % This class contains methods for evaluating the performance of a classification model. It stores the
    % following properties:
    %
    %   modelId:        A string identifying the model.
    %   yPred:          The predicted labels produced by the model.
    %   accuracy:       The proportion of correct predictions made by the model.
    %   precision:      The proportion of correct positive predictions made by the model.
    %   recall:         The proportion of actual positive cases that were correctly predicted by the model.
    %   f1_score:       The F1 score, which is a measure of the balance between precision and recall.
    %   confusionMatrix: A matrix showing the number of true positive, false positive, false negative, and true negative predictions made by the model.
    %
    % The `Classifier` class has the following methods:
    %
    %   Classifier:       Constructor for creating a `Classifier` object.
    %   getMetrics:       Static method for calculating various performance metrics for a model.
    %   findHighestAccuracy: Static method for finding the `Classifier` object with the highest accuracy among a list of `Classifier` objects.
    %   printScores:      Method for printing the accuracy, precision, recall, and F1 score of a `Classifier` object.


    properties
        modelId
        model
        yPred
        accuracy
        precision
        recall
        f1_score
        confusionMatrix
    end

    methods(Static)

        function [accuracy, precision, recall, f1_score, confusionMatrix] = getMetrics(model, yPred, yObserved)
            % getMetrics
            %
            % Calculates various performance metrics for a classification model.
            %
            % Syntax:
            %   [accuracy, resubError, precision, recall, f1_score, confusionMatrix] = getMetrics(model, yPred, yObserved)
            %
            % Inputs:
            %   model:       A trained classification model.
            %   yPred:       The predicted labels produced by the model.
            %   yObserved:   The true labels.
            %
            % Outputs:
            %   accuracy:    The proportion of correct predictions made by the model.
            %   resubError:  The mean squared error on the resubstitution of the data into the trained model.
            %   precision:   The proportion of correct positive predictions made by the model.
            %   recall:      The proportion of actual positive cases that were correctly predicted by the model.
            %   f1_score:    The F1 score, which is a measure of the balance between precision and recall.
            %   confusionMatrix: A matrix showing the number of true positive, false positive, false negative, and true negative predictions made by the model.

            % calculating confusion matrix
            truePositives = sum((yPred == 1) & (yObserved == 1));
            falsePositives = sum((yPred == 1) & (yObserved == 0));
            falseNegatives = sum((yPred == 0) & (yObserved == 1));

            correctPredictions = (yPred == yObserved);
            isMissing = isnan(yObserved);
            correctPredictions = correctPredictions(~isMissing);
            accuracy = sum(correctPredictions)/length(correctPredictions);

            precision = truePositives / (truePositives + falsePositives);
            recall = truePositives / (truePositives + falseNegatives);
            f1_score = (2 * precision * recall) / (precision + recall);
            confusionMatrix = confusionmat(yObserved, yPred);

        end

        function bestClassifier = findHighestAccuracy(models)

            highestAccuracy = 0;

            for i = 1:2
                if models(i).accuracy > highestAccuracy
                    highestAccuracy = models(i).accuracy;
                    bestClassifier = models(i);
                end
            end
        end
    end

    methods

        function obj = Classifier(modelId, model, yPred, accuracy, precision, recall, f1_score, confusionMatrix)

            if nargin > 0
                obj.modelId = modelId;
                obj.model = model;
                obj.yPred = yPred;
                obj.accuracy= accuracy;
                obj.precision = precision;
                obj.recall = recall;
                obj.f1_score = f1_score;
                obj.confusionMatrix = confusionMatrix;
            end
        end

        function [] = printScores(obj)
            % printScores
            %
            % Prints the accuracy, precision, recall, and F1 score of a `Classifier` object.
            %
            % Syntax:
            %   printScores(obj)
            %
            % Inputs:
            %   obj:       A `Classifier` object.
            %
            % Outputs:
            %   None, displays the accuracy, precision, recall, and F1 score on the command window.

            fprintf("Accuracy: %.4f\n", obj.accuracy);
            fprintf("Precision: %.4f\n", obj.precision);
            fprintf("Recall: %.4f\n", obj.recall);
            fprintf("F1 score: %.4f\n", obj.f1_score);
        end
    end
end
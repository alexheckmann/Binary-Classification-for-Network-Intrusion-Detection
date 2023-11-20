classdef RandomForestClassifier < Classifier
    % RandomForestClassifier is a data structure for storing and evaluating a random forest classifier 
    % since Matlab does not provide something suitable for this task.
    %
    % This class extends the Classifier class and adds a property for storing the random forest model.
    %
    % Properties:
    %   randomForest: a model object storing the random forest model
    %
    % Methods:
    %   RandomForestClassifier: constructor for the RandomForestClassifier class
    %
    %   The RandomForestClassifier constructor takes in the following arguments:
    %
    %   randomForest: a model object storing the random forest model
    %   modelId: a string identifier for the model
    %   yPred: a vector of predicted class labels
    %   yObserved: a vector of observed class labels
    %
    %   The constructor calls the parent class's constructor to set the modelId, yPred, accuracy,
    %   precision, recall, f1_score, and confusionMatrix properties. It also sets the randomForest
    %   property using the input random forest model object.
    %
    properties
        randomForest
    end

    methods

        function obj = RandomForestClassifier(randomForest, modelId, yPred, yObserved)

            [accuracy, precision, recall, f1_score, confusionMatrix] = Classifier.getMetrics(randomForest, yPred, yObserved);
            obj = obj@Classifier(modelId, yPred, accuracy, precision, recall, f1_score, confusionMatrix);
            obj.randomForest = randomForest;
        end
    end
end
classdef DecisionTreeClassifier < Classifier
    % DecisionTreeClassifier is a data structure for storing and evaluating a decision tree classifier 
    % since Matlab does not provide something suitable for this task.
    %
    % This class extends the Classifier class and adds a property for storing the classification tree.
    %
    % Properties:
    %   classificationTree: a tree object storing the classification tree
    %
    % Methods:
    %   DecisionTreeClassifier: constructor for the DecisionTreeClassifier class
    %
    %   The DecisionTreeClassifier constructor takes in the following arguments:
    %
    %   classificationTree: a tree object storing the classification tree
    %   modelId: a string identifier for the model
    %   yPred: a vector of predicted class labels
    %   yObserved: a vector of observed class labels
    %
    %   The constructor calls the parent class's constructor to set the modelId, yPred, accuracy,
    %   precision, recall, f1_score, and confusionMatrix properties. It also sets the classificationTree
    %   property using the input classification tree object.
    %
    properties
        classificationTree
    end

    methods

        function obj = DecisionTreeClassifier(classificationTree, modelId, yPred, yObserved)

            [accuracy, precision, recall, f1_score, confusionMatrix] = Classifier.getMetrics(classificationTree, yPred, yObserved);
            obj = obj@Classifier(modelId, yPred, accuracy, precision, recall, f1_score, confusionMatrix);
            obj.classificationTree = classificationTree;
        end
    end
end
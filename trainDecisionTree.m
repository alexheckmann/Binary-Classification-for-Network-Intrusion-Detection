function [trainedDecisionTree] = trainDecisionTree(classificationData)
    dataset = classificationData.dataset;
    predictors = dataset(:, classificationData.predictors);
    response = classificationData.response;
    isCategoricalPredictor = classificationData.isCategoricalPredictor;
    
end
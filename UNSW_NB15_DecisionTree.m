%% clean up

close gcf
% closes all figures
close all
% clears the workspace
clear all
% clears the command window
clc

%% Setting up data

% setting rng() value for reproducibility
rng(42);
train_df_original = readtable("data\UNSW_NB15_training.csv");
test_df_original = readtable("data\UNSW_NB15_testing.csv");
[train_df, test_df] = resetData(train_df_original, test_df_original);
% can be set here since train_df_original.label == test_df.label at all
% times, no rows will be removed later
yObserved = test_df_original.label;
numericalPredictors = [2, 6:(size(train_df_original) - 2)];
cost_function = [0 1; 10 0];
disp("Loaded data...")

%% DT Model 1: Original model

[train_df, test_df] = resetData(train_df_original, test_df_original);

model1 = fitctree(train_df, "label");
yPred = predict(model1, test_df);
% Accuracy: 0.86282
% Recall: 0.9581
model1 = DecisionTreeClassifier(model1, "dt", yPred, yObserved);
model1.printScores()

%% DT Model 2: adjusted cost

[train_df, test_df] = resetData(train_df_original, test_df_original);

% source: Matlab fitctree documentation
% increasing cost for false negatives
% this is due to the higher cost of security breaches
model2 = fitctree(train_df, "label", "Cost", cost_function);
yPred = predict(model2, test_df);
% accuracy: 0.85495 // 0.0787 decrease compared to best model, might be
% considerable optimization due to less false negatives which are way
% more costly than false positives
% Recall: 0.9706
%
model2 = DecisionTreeClassifier(model2, "dt w/ adjusted cost", yPred, yObserved);
model2.printScores()

%% Feature selection: Plotting correlation matrix
[train_df, test_df] = resetData(train_df_original, test_df_original);

correlationMatrix = getCorrelation(train_df, numericalPredictors);
xvalues = train_df.Properties.VariableNames(:, numericalPredictors);
yvalues = train_df.Properties.VariableNames(:, numericalPredictors);
corrheatmap(correlationMatrix, xvalues, yvalues);

%% DT Model 3: Feature selection & adjusted cost

[train_df, test_df] = resetData(train_df_original, test_df_original);
% Removing features w/ high correlation (roughly 0.75).
% Features can be removed because other network request features
% perfectly describe them. E.g. loss and bytes are already described by the
% number of network packets for source and destination, respectively.
removedFeatures = ["sloss", "dloss", "dttl", "dbytes", "swin", "synack", ...
    "dwin", "tcprtt", "ct_srv_dst", "ct_srv_src", "ct_dst_ltm", ...
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", ...
    "is_ftp_login", "ct_src_ltm", "ct_srv_dst"];
[train_df, test_df, numericalPredictors, ~] = cleanData(train_df, test_df, removedFeatures);

model3 = fitctree(train_df, "label", "Cost", cost_function);
yPred = predict(model3, test_df);
% Accuracy: 0.84997
% Recall: 0.9650
model3 = DecisionTreeClassifier(model3, "dt w/o high correlation features", yPred, yObserved);
model3.printScores()

%% DT Model 4: feature selection, adjusted cost & selection curvature

[train_df, test_df] = resetData(train_df_original, test_df_original);
removedFeatures = ["sloss", "dloss", "dttl", "dbytes", "swin", "synack", ...
    "dwin", "tcprtt", "ct_srv_dst", "ct_srv_src", "ct_dst_ltm", ...
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", ...
    "is_ftp_login", "ct_src_ltm", "ct_srv_dst"];
[train_df, test_df, ~] = cleanData(train_df, test_df, removedFeatures);

model4 = fitctree(train_df, "label", "Cost", cost_function, "PredictorSelection", "curvature");
yPred = predict(model4, test_df);
% Accuracy: 0.8497 // improvement compared to "dt 1" & less complexity
% Recall: 0.9658
model4 = DecisionTreeClassifier(model4, "dt w/o high correlation features & adjusted cost", yPred, yObserved)
model4.printScores()

%% DT Model 5: feature selection, adjusted cost & selection interaction-curvature

[train_df, test_df] = resetData(train_df_original, test_df_original);
removedFeatures = ["sloss", "dloss", "dttl", "dbytes", "swin", "synack", ...
    "dwin", "tcprtt", "ct_srv_dst", "ct_srv_src", "ct_dst_ltm", ...
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", ...
    "is_ftp_login", "ct_src_ltm", "ct_srv_dst"];
[train_df, test_df, ~] = cleanData(train_df, test_df, removedFeatures);

model5 = fitctree(train_df, "label",  "Cost", cost_function, "PredictorSelection", "interaction-curvature");
yPred = predict(model5, test_df);
% Accuracy: 0.8497 // no improvement in accuracy, therefore default or
% interaction-curvature option will be used
% Recall: 0.9658
model5 = DecisionTreeClassifier(model5, "dt w/ predictor selection: interaction curvature", yPred, yObserved);
model5.printScores()

%% DT Model 6: feature selection, adjusted cost & 100 splits

[train_df, test_df] = resetData(train_df_original, test_df_original);
removedFeatures = ["sloss", "dloss", "dttl", "dbytes", "swin", "synack", ...
    "dwin", "tcprtt", "ct_srv_dst", "ct_srv_src", "ct_dst_ltm", ...
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", ...
    "is_ftp_login", "ct_src_ltm", "ct_srv_dst"];
[train_df, test_df, ~] = cleanData(train_df, test_df, removedFeatures);

model6 = fitctree(train_df, "label",  "Cost", cost_function, 'MaxNumSplits', 100);
yPred = predict(model6, test_df);
% Accuracy: 0.8113 // no improvement in accuracy, therefore default or
% interaction-curvature option will be used
% Recall: 0.9993
model6 = DecisionTreeClassifier(model6, "dt w/ 100 splits", yPred, yObserved);
model6.printScores()

%% DT Model 7: feature selection, adjusted cost & selection interaction-curvature

[train_df, test_df] = resetData(train_df_original, test_df_original);
removedFeatures = ["sloss", "dloss", "dttl", "dbytes", "swin", "synack", ...
    "dwin", "tcprtt", "ct_srv_dst", "ct_srv_src", "ct_dst_ltm", ...
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", ...
    "is_ftp_login", "ct_src_ltm", "ct_srv_dst"];
[train_df, test_df, ~] = cleanData(train_df, test_df, removedFeatures);

model7 = fitctree(train_df, "label",  "Cost", cost_function, 'MaxNumSplits', 50);
yPred = predict(model7, test_df);
% Accuracy: 0.8108 // no improvement in accuracy, therefore default or
% interaction-curvature option will be used
% Recall: 0.9999
model7 = DecisionTreeClassifier(model7, "dt w/ predictor selection: interaction curvature", yPred, yObserved);
model7.printScores()

%% DT Model 8: feature selection & gridsearch

[train_df, test_df] = resetData(train_df_original, test_df_original);
removedFeatures = ["sloss", "dloss", "dttl", "dbytes", "swin", "synack", ...
    "dwin", "tcprtt", "ct_srv_dst", "ct_srv_src", "ct_dst_ltm", ...
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", ...
    "is_ftp_login", "ct_src_ltm", "ct_srv_dst"];
[train_df, test_df, ~] = cleanData(train_df, test_df, removedFeatures);
tic
model7 = fitctree(train_df, "label", "OptimizeHyperparameters", "all", ...
    "Cost", cost_function, ...
    "HyperparameterOptimizationOptions", struct('Optimizer', 'gridsearch'));
rf_time_elapsed = toc
yPred = predict(model7, test_df);
save("UNSW-NB15 DT.mat", "model7")
% accuracy: 0.8736
% recall: 0.9712
model7 = DecisionTreeClassifier(model7, "dt w/ optimization all", yPred, yObserved)
model7.printScores()

%% RF Model 0: simple implementation

[train_df, test_df] = resetData(train_df_original, test_df_original);
x_train = train_df;
x_train(:, "label") = [];
y_train = train_df.label;
rf_learner = templateTree("Reproducible",true)

rf_model1 = fitcensemble(x_train, y_train, "Learners", rf_learner, ...
    "CategoricalPredictors",{'proto', 'service', 'state'});
yPred = predict(rf_model1, test_df);

rf_model1 = RandomForestClassifier(rf_model1, "rf", yPred, yObserved);
% accuracy: 0.8745
% recall: 0.9806
rf_model1.printScores()

%% RF Model 1: cost function

[train_df, test_df] = resetData(train_df_original, test_df_original);
x_train = train_df;
x_train(:, "label") = [];
y_train = train_df.label;
rf_learner = templateTree("Reproducible",true)

rf_model1 = fitcensemble(x_train, y_train, "Learners", rf_learner, ...
    "CategoricalPredictors",{'proto', 'service', 'state'}, ...
    "Cost", cost_function);
yPred = predict(rf_model1, test_df);

rf_model1 = RandomForestClassifier(rf_model1, "rf w/ cost", yPred, yObserved);
% accuracy: 0.8779
% recall: 0.9823
rf_model1.printScores()

%% RF Model 3: feature selection

[train_df, test_df] = resetData(train_df_original, test_df_original);
removedFeatures = ["sloss", "dloss", "dttl", "dbytes", "swin", "synack", ...
    "dwin", "tcprtt", "ct_srv_dst", "ct_srv_src", "ct_dst_ltm", ...
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", ...
    "is_ftp_login", "ct_src_ltm", "ct_srv_dst"];
[train_df, test_df, ~] = cleanData(train_df, test_df, removedFeatures);
x_train = train_df;
x_train(:, "label") = [];
y_train = train_df.label;
rf_learner = templateTree("Reproducible",true);

rf_model3 = fitcensemble(x_train, y_train, "Learners", rf_learner, ...
    "CategoricalPredictors",{'proto', 'service', 'state'}, ...
    "Cost", cost_function);
yPred = predict(rf_model2, test_df);

rf_model3 = RandomForestClassifier(rf_model2, "rf w/ feature selection", yPred, yObserved);
% accuracy: 0.8701
% recall: 0.9737
rf_model3.printScores()

%% RF Model 3: feature selection & bagging

[train_df, test_df] = resetData(train_df_original, test_df_original);
removedFeatures = ["sloss", "dloss", "dttl", "dbytes", "swin", "synack", ...
    "dwin", "tcprtt", "ct_srv_dst", "ct_srv_src", "ct_dst_ltm", ...
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", ...
    "is_ftp_login", "ct_src_ltm", "ct_srv_dst"];
[train_df, test_df, ~] = cleanData(train_df, test_df, removedFeatures);
x_train = train_df;
x_train(:, "label") = [];
y_train = train_df.label;
tic
rf_learner = templateTree("Reproducible",true);

rf_model3 = fitcensemble(x_train, y_train, "Learners", rf_learner, ...
    "CategoricalPredictors",{'proto', 'service', 'state'}, ...
    "Method","Bag", ...
    "Cost", cost_function);
rf_time_elapsed = toc
save("UNSW-NB15 RF.mat", "rf_model3")
yPred = predict(rf_model3, test_df);

rf_model3 = RandomForestClassifier(rf_model3, "rf w/ feature selection & bagging", yPred, yObserved);
% accuracy: 0.8115
% recall: 0.9995
rf_model3.printScores()

%% RF Model 4: bagging

[train_df, test_df] = resetData(train_df_original, test_df_original);
x_train = train_df;
x_train(:, "label") = [];
y_train = train_df.label;
rf_learner = templateTree("Reproducible",true);

rf_model4 = fitcensemble(x_train, y_train, "Learners", rf_learner, ...
    "CategoricalPredictors",{'proto', 'service', 'state'}, ...
    "Method","Bag", ...
    "Cost", cost_function);
yPred = predict(rf_model4, test_df);

rf_model4 = RandomForestClassifier(rf_model4, "rf w/ bagging", yPred, yObserved);
% accuracy: 0.8119
% recall: 0.9989
rf_model4.printScores()

%% RF Model 5: adaptive boosting

[train_df, test_df] = resetData(train_df_original, test_df_original);
x_train = train_df;
x_train(:, "label") = [];
y_train = train_df.label;
rf_learner = templateTree("Reproducible",true);

rf_model5 = fitcensemble(x_train, y_train, "Learners", rf_learner, ...
    "CategoricalPredictors",{'proto', 'service', 'state'}, ...
    "Method","AdaBoostM1", ...
    "Cost", cost_function);
save("UNSW-NB15 RF.mat", "rf_model5")
yPred = predict(rf_model5, test_df);

rf_model5 = RandomForestClassifier(rf_model5, "rf w/ adaboost", yPred, yObserved);
% accuracy: 0.8102
% recall: 1.0000
rf_model5.printScores()

%% RF Model 6: adaptive boosting w/o cost function

[train_df, test_df] = resetData(train_df_original, test_df_original);
x_train = train_df;
x_train(:, "label") = [];
y_train = train_df.label;
rf_learner = templateTree("Reproducible",true);

rf_model6 = fitcensemble(x_train, y_train, "Learners", rf_learner, ...
    "CategoricalPredictors",{'proto', 'service', 'state'}, ...
    "Method","AdaBoostM1");
yPred = predict(rf_model6, test_df);

rf_model6 = RandomForestClassifier(rf_model6, "rf w/ adaboost w/o cost", yPred, yObserved);
% accuracy: 0.8760
% recall: 0.9880
rf_model6.printScores()
%% RF Model 7: LearnRate & MaxNumSplits optimization

[train_df, test_df] = resetData(train_df_original, test_df_original);
removedFeatures = ["sloss", "dloss", "dttl", "dbytes", "swin", "synack", ...
    "dwin", "tcprtt", "ct_srv_dst", "ct_srv_src", "ct_dst_ltm", ...
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", ...
    "is_ftp_login", "ct_src_ltm", "ct_srv_dst"];
[train_df, test_df, ~] = cleanData(train_df, test_df, removedFeatures);
x_train = train_df;
x_train(:, "label") = [];
y_train = train_df.label;
tic
rf_learner = templateTree("Reproducible",true)
% Train a random forest using the fitcensemble function
% and optimize the hyperparameters using the OptimizeHyperparameters function
rf_model7 = fitcensemble(x_train, y_train, ...
    'NumBins', 20, ...
    'NumLearningCycles', 25, ...
    'OptimizeHyperparameters',{'LearnRate','MaxNumSplits'}, ...
    'Learners', rf_learner, ...
    "CategoricalPredictors",{'proto', 'service', 'state'}, ...
    "Cost", cost_function, ...
    'HyperparameterOptimizationOptions', ...
struct('AcquisitionFunctionName','expected-improvement-plus'));
rf_time_elapsed = toc
yPred = predict(rf_model7, test_df);
% Accuracy: 0.8763 // no improvement in accuracy
rf_model7 = RandomForestClassifier(rf_model7, "rf w/ optimization all", yPred, yObserved)
% accuracy: 0.8540
% recall: 0.9786
rf_model7.printScores()

%%
% source: Matlab documentation "confusionchart"
chart_dt = confusionchart(model7.confusionMatrix,'RowSummary','row-normalized','ColumnSummary','column-normalized', 'Title', "Decision Tree");
chart_rf = confusionchart(rf_model5.confusionMatrix,'RowSummary','row-normalized','ColumnSummary','column-normalized', 'Title', "Random Forest");

x = categorical({'Accuracy', 'Recall', 'Precision', 'F1 score'})
metrics = [0.8108, 0.8102
1, 1
0.7442, 0.7437
0.8534, 0.8530

]

b = bar(x, metrics, Title, "Decision Tree vs Random Forest")

xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
labels2 = string(b(2).YData);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

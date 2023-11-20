close gcf
% closes all figures
close all
% clears the workspace
clear all
% clears the command window
clc

% Code was written using Matlab 2022b.
% using load() returns a table in Matlab 2022 instead of the actual
% workspace variable.
% if the initial line already loaded the correct variable, delete the
% second ones with the reassignment
test_df_dt = load("test_set.mat");
test_df_dt = test_df_dt.test_df;

dt = load("UNSW-NB15 DT.mat");
dt = dt.model7;
y_pred_dt = predict(dt, test_df_dt);
[accuracy_dt, precision_dt, recall_dt, f1_score_dt, confusionMatrix_dt] = Classifier.getMetrics(dt, y_pred_dt, test_df_dt.label)


test_df_rf = load("test_set_rf.mat");
test_df_rf = test_df_rf.test_df;
rf = load("UNSW-NB15 RF.mat");
rf = rf.rf_model5;
y_pred_rf = predict(rf, test_df_rf);
[accuracy_rf, precision_rf, recall_rf, f1_score_rf, confusionMatrix_rf] = Classifier.getMetrics(rf, y_pred_rf, test_df_rf.label)

%% code to create figures

% source: Matlab documentation "confusionchart"
chart_dt = confusionchart(confusionMatrix_dt,'RowSummary','row-normalized','ColumnSummary','column-normalized', 'Title', "Decision Tree");
chart_rf = confusionchart(confusionMatrix_rf,'RowSummary','row-normalized','ColumnSummary','column-normalized', 'Title', "Random Forest");

x = categorical({'Accuracy', 'Recall', 'Precision', 'F1 score'})
% left value is dt, right one is rf
metrics = [0.8108, 0.8102
1, 1
0.7442, 0.7437
0.8534, 0.8530
]

b = bar(x, metrics);

xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
labels2 = string(b(2).YData);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

%% code to train the best decision tree classifier
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

%% code to run the best random forest classifier
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
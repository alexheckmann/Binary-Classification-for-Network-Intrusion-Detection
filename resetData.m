function [train_df, test_df] = resetData(train_df_original, test_df_original)
    % resetData
    %
    % Resets the train and test data to their original state by performing the following operations:
    %   - Copies the original train and test data to separate dataframes
    %   - Converts string variables to categorical data
    %   - Removes the "id" and "attack_cat" features
    %
    % Syntax:
    %   [train_df, test_df] = resetData(train_df_original, test_df_original)
    %
    % Inputs:
    %   train_df_original:  Original train data.
    %   test_df_original:   Original test data.
    %
    % Outputs:
    %   train_df:           Processed train data.
    %   test_df:            Processed test data.

    % copying train and test data to work on seperate copy
    train_df = train_df_original;
    test_df = test_df_original;

    % converts string to categorical data
    train_df = convertvars(train_df, @isstring, "categorical");
    test_df = convertvars(test_df, @isstring, "categorical");

    % removing id: just an ordered index of all records
    % removing attack_cat: label for multi class classification
    removedFeatures = ["id", "attack_cat"];
    [train_df, test_df] = cleanData(train_df, test_df, removedFeatures);

    disp("Reset data to default state...");

end

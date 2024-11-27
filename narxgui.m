function narx_gui()
% GUI for Time Series Forecasting with a NARX Model
% -------------------------------------------------
% This tool uses a feedforward neural network to forecast a time series (y) 
% based on a set of exogenous regressors (X).
% 
% Author: Foued Saadaoui, 2024
% 
% Instructions:
% - Prepare a .mat file containing:
%   + A time series vector (y).
%   + A matrix of exogenous regressors (X) with the same number of observations as y.
% - Load the data through the GUI and configure model parameters as needed.

    
    hFig = figure('Position', [300, 300, 400, 300], 'MenuBar', 'none', ...
        'Name', 'NARX Neural Network GUI', 'NumberTitle', 'off', 'Resize', 'off');
    
    uicontrol('Style', 'pushbutton', 'Position', [50, 220, 300, 40], ...
        'String', 'Import Data (y and X)', ...
        'Callback', @import_data);
    
    %--------------------------- Input for lag order ---------------------
    uicontrol('Style', 'text', 'Position', [50, 180, 120, 20], ...
        'String', 'Lag Order:', 'HorizontalAlignment', 'left');
    hLag = uicontrol('Style', 'edit', 'Position', [180, 180, 170, 25], ...
        'String', '1');
    
    %--------------------------- Input for number of neurons -------------
    uicontrol('Style', 'text', 'Position', [50, 140, 120, 20], ...
        'String', 'Hidden Neurons:', 'HorizontalAlignment', 'left');
    hNeurons = uicontrol('Style', 'edit', 'Position', [180, 140, 170, 25], ...
        'String', '10');
    
    %--------------------------- Train button ----------------------------
    uicontrol('Style', 'pushbutton', 'Position', [50, 80, 300, 40], ...
        'String', 'Train and Evaluate NARX Network', ...
        'Callback', @(~, ~) train_network(str2double(hLag.String), str2double(hNeurons.String)));
    
    %--------------------------- Placeholder for storing data ------------
    hFig.UserData.y = [];
    hFig.UserData.X = [];

    % Callback to import data
    function import_data(~, ~)
        [file, path] = uigetfile('*.mat', 'Select a MAT File');
        if isequal(file, 0)
            msgbox('No file selected', 'Error', 'error');
            return;
        end
        data = load(fullfile(path, file));
        if isfield(data, 'y') && isfield(data, 'X')
            hFig.UserData.y = data.y;
            hFig.UserData.X = data.X;
            msgbox('Data successfully loaded!', 'Success');
        else
            msgbox('The file must contain variables "y" and "X".', 'Error', 'error');
        end
    end

    % Callback to train network
    function train_network(lag_order, num_hidden_neurons)
        if isempty(hFig.UserData.y) || isempty(hFig.UserData.X)
            msgbox('Please load the data first.', 'Error', 'error');
            return;
        end
        
        y = hFig.UserData.y;
        X = hFig.UserData.X;

        %--------------------------- Validate inputs ---------------------
        if isnan(lag_order) || lag_order <= 0
            msgbox('Lag Order must be a positive integer.', 'Error', 'error');
            return;
        end
        if isnan(num_hidden_neurons) || num_hidden_neurons <= 0
            msgbox('Number of Hidden Neurons must be a positive integer.', 'Error', 'error');
            return;
        end

        %--------------------------- Create lagged inputs ----------------
        [num_samples, num_features] = size(X);
        y_lagged = [];
        X_lagged = [];
        
        for lag = 1:lag_order
            y_lagged = [y_lagged, [nan(lag, 1); y(1:end-lag)]];
            X_lagged = [X_lagged, [nan(lag, num_features); X(1:end-lag, :)]];
        end
        
        valid_indices = ~any(isnan([y_lagged, X_lagged, y]), 2);
        X_lagged = X_lagged(valid_indices, :);
        y_lagged = y_lagged(valid_indices, :);
        y_target = y(valid_indices);
        
        inputs = [y_lagged, X_lagged]';
        targets = y_target';

        % Data splitting
        [trainInd, valInd, testInd] = divideind(length(y_target), ...
            1:floor(0.8*length(y_target)), ...
            floor(0.8*length(y_target))+1:floor(0.9*length(y_target)), ...
            floor(0.9*length(y_target))+1:length(y_target));

        % Neural network
        net = feedforwardnet(num_hidden_neurons);
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = trainInd;
        net.divideParam.valInd = valInd;
        net.divideParam.testInd = testInd;

        % Train network
        [net, tr] = train(net, inputs, targets);

        % Evaluate
        predictions = net(inputs);
        mse_train = mse(net, targets(tr.trainInd), predictions(tr.trainInd));
        mse_val = mse(net, targets(tr.valInd), predictions(tr.valInd));
        mse_test = mse(net, targets(tr.testInd), predictions(tr.testInd));

        %--------------------------- Display results ---------------------
        msgbox(sprintf('Training MSE: %.4f\nValidation MSE: %.4f\nTesting MSE: %.4f', ...
            mse_train, mse_val, mse_test), 'Results');

        %--------------------------- Plot --------------------------------
        figure;
        plot(targets, 'b', 'DisplayName', 'Actual');
        hold on;
        plot(predictions, 'r', 'DisplayName', 'Predicted');
        legend;
        title('Actual vs Predicted Values');
        xlabel('Time');
        ylabel('Output');
        grid on;
    end
end

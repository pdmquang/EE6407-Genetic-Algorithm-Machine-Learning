function [Mu, Sigma, Prior] = BayesDecision_Parameters(X, y)
    numFeatures = size(X, 2);
    classes = unique(y);
    numClasses = length(classes);

    Mu = zeros(numClasses, numFeatures); 
    Sigma = zeros(size(X, 2), size(X, 2), numClasses); % Covariance matrices
    Prior = zeros(numClasses, 1); % Prior probabilities
    
    for i = 1:numClasses
        idx = find(y==classes(i));
        Mu(i, :) = mean(X(idx,:)); % mean of features for class i
        Sigma(:, :, i) = cov(X(idx, :)); % Covariance matrix for class i 
        Prior(i) = sum(idx) / length(y); % Prior probability or class i
    end
end

function y_preds = BayesDecision_predict(X, Classes, Mu, Sigma, Prior)
    numClasses = length(Classes);

    % Function with (x, mu, sigma) as parameter
    gaussianPdf = @(x, mu, sigma) ...
        (1 / sqrt((2 * pi) ^ size(x, 2) * det(sigma))) * exp(-0.5 * ((x - mu) / sigma) * (x - mu)');
    
    % Posterior probability is a scalar value for binary classification
    % represents the probability of a single class
    y_preds = zeros(numel(X(:, 1)), 1);
    
    for i = 1:length(X)
        Posteriors = zeros(numClasses, 1);
        features = X(i, :);

        for j = 1:numClasses
            likelihood = gaussianPdf(features, Mu(j, :), Sigma(:, :, j));
            Posteriors(j) = Prior(j) * likelihood; % Compute posterior P(Y|X)
        end
        
        [~, class_i] = max(Posteriors);
        y_preds(i) = Classes(class_i);
    end
end

Classes = unique(y);
numClasses = length(unique(y));
X_input = X;
% X_input = X_filled;
% X_input = X_zscore;

[Mu, Sigma, Prior] = BayesDecision_Parameters(X_input, y);
ypreds_Decision = BayesDecision_predict(X_test, Classes, Mu, Sigma, Prior);


% N_Folds = [2, 5, 10];
% BayesDecision_Scores = zeros(numel(N_Folds), 1);
% 
% rng(1);
% 
% for n = 1:numel(N_Folds)
%     cv = cvpartition(y, 'KFold', N_Folds(n));
%     accuracies = zeros(cv.NumTestSets, 1);
% 
%     for i = 1:cv.NumTestSets
%         % Get train and test indices
%         trainIdx = cv.training(i);
%         testIdx = cv.test(i);
% 
%         % Prepare training and test data
%         X_train = X_input(trainIdx, :);
%         y_train = y(trainIdx);
% 
%         X_test = X_input(testIdx, :);
%         y_test = y(testIdx);
% 
%         [Mu, Sigma, Prior] = BayesDecision_Parameters(X_train, y_train);
% 
%         y_predicted = zeros(numel(y_test),1);
%         for j = 1:length(y_test)
%             predictedClass = bayesDecisionRule(X_test(j, :), Classes, Mu, Sigma, Prior);
%             y_predicted(j) = predictedClass;
%         end
% 
%         % Calculate accuracy for this fold
%         predictions = y_test == y_predicted;
%         accuracies(i) = sum(predictions) / numel(predictions);
%     end
% 
%     BayesDecision_Scores(n) = mean(accuracies);
% end
% 
% 
% % disp('Finish...');
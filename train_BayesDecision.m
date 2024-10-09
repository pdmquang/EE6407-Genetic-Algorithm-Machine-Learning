% size(X, 2) => (X, dim=2) => no of cols = 5]
% numFeatures = size(X, 2);
% classes = unique(y);
% numClasses = length(classes);

function [Mu, Sigma, Prior] = BayesDecision_Parameters(X, y)
    numFeatures = size(X, 2);
    classes = unique(y);
    numClasses = length(classes);

    Mu = zeros(numClasses, numFeatures); 
    Sigma = zeros(numClasses, numFeatures);
    Prior = zeros(numClasses, 1); % Prior probabilities
    
    for i = 1:numClasses
        idx = find(y==classes(i));
        Mu(i, :) = mean(X(idx,:)); % mean of features for class i
        Sigma(i, :) = std(X(idx,:)); % Covariance matrix for class i 
        Prior(i) = sum(idx) / length(y); % Prior probability or class i
    end
end

function predictedClass = bayesDecisionRule(X, numClasses, Mu, Sigma, Prior)
    Posteriors = zeros(numClasses, length(X));
    likelihood =  normpdf(X, Mu, Sigma);

    for i = 1:numClasses
        Posteriors(i, :) = Prior(i) * likelihood(i, :); % Compute posterior P(Y|X)
    end
    % [Matrix, Index]
    [~, predictedClassByFeature] = max(Posteriors); % Classify as the class with max posterior probability
    predictedClass = mode(predictedClassByFeature);
end

numClasses = 2;

% Test classification on a new data point
[Mu, Sigma, Prior] = BayesDecision_Parameters(X, y);
predictedClass = bayesDecisionRule(X(4, :), numClasses, Mu, Sigma, Prior);
disp(['Predicted class: ', num2str(predictedClass) ]);

% rng(1);
% cv = cvpartition(y, 'KFold', 10);
% accuracies = zeros(cv.NumTestSets, 1);
% 
% for i = 1:cv.NumTestSets
%     % Get train and test indices
%     trainIdx = cv.training(i);
%     testIdx = cv.test(i);
% 
%     % Prepare training and test data
%     X_train = X(trainIdx, :);
%     y_train = y(trainIdx);
% 
%     X_test = X(testIdx, :);
%     y_test = y(testIdx);
% 
%     [Mu, Sigma, Prior] = BayesDecision_Parameters(X_train, y_train);
%     predictedClass = bayesDecisionRule(X_test, y_test, Mu, Sigma, Prior);
% 
%     true_classes = unique(Y_train);
%     % Calculate accuracy for this fold
%     accuracies(i) = sum(find(Y_test, true_classes(predictedClass))) / length(Y_test);
% 
% end

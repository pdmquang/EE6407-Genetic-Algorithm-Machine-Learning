% size(X, 2) => (X, dim=2) => no of cols = 5]
numFeatures = size(X, 2);

Mu = zeros(numClasses, numFeatures); 
Sigma = zeros(numClasses, numFeatures);
Prior = zeros(numClasses, 1); % Prior probabilities

for i = 1:numClasses
    idx = find(y==classes(i));
    Mu(i, :) = mean(X(idx,:)); % mean of features for class i
    Sigma(i, :) = std(X(idx,:)); % Covariance matrix for class i 
    Prior(i) = sum(idx) / length(y); % Prior probability or class i
end

function predictedClass = bayesDecisionRule(X_new, Mu, Sigma, Prior, numClasses)
    Posteriors = zeros(numClasses, length(X_new));
    likelihood =  normpdf(X_new, Mu, Sigma);

    for i = 1:numClasses
        Posteriors(i, :) = Prior(i) * likelihood(i, :); % Compute posterior P(Y|X)
    end
    % [Matrix, Index]
    [~, predictedClass] = max(Posteriors); % Classify as the class with max posterior probability
end

% Test classification on a new data point
newDataPoint = [5.0, 3.5, 1.5, 0.2, 0.5]; % Example data point
predictedClassIndex = bayesDecisionRule(newDataPoint, Mu, Sigma, Prior, numClasses);

disp(['Predicted class: ', num2str(classes(predictedClassIndex(:,1))) ]);
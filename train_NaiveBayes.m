% https://www.mathworks.com/help/stats/fitcnb.html
NaiveBayes_Mdl = fitcnb(X, label, 'ClassNames', {'-1', '1'});

% [labels,PostProbs,MisClassCost] = predict(NaiveBayes_Mdl, newDataPoint);

rng(1); % For reproducibility
NaiveBayes_CVMdl = crossval(NaiveBayes_Mdl);
NaiveBayes_Loss = kfoldLoss(NaiveBayes_CVMdl);

NaiveBayes_Avg_Accuracy = 1 - NaiveBayes_Loss; % Accuracy is 1 minus the classification error
disp(['Accuracy: ', num2str(NaiveBayes_Avg_Accuracy)]);
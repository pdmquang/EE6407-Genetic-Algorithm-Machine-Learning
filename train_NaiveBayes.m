% % https://www.mathworks.com/help/stats/fitcnb.html
NaiveBayes_Mdl = fitcnb(X, label, 'ClassNames', {'-1', '1'});
ypreds_Naive = predict(NaiveBayes_Mdl,X_test);
ypreds_Naive = str2double(ypreds_Naive);

% NaiveBayes_Mdl = fitcnb(X_filled, label, 'ClassNames', {'-1', '1'});
% NaiveBayes_Mdl = fitcnb(X_zscore, label, 'ClassNames', {'-1', '1'});

% Params = cell2mat(NaiveBayes_Mdl.DistributionParameters);
% Naive_Mu = Params(2*(1:2)-1, 1:size(X, 2)); % Means
% Naive_Std = Params(2*(1:2), 1:size(X, 2)); % Standard Deviation
% 
% Params = cell2mat(NaiveBayes_Mdl_filled.DistributionParameters);
% Naive_Mu_filled = Params(2*(1:2)-1, 1:size(X, 2)); % Means
% Naive_Std_filled = Params(2*(1:2), 1:size(X, 2)); % Standard Deviation

% rng(1); % For reproducibility
% 
% N_Folds = [2, 5, 10];
% NaiveBayes_Scores = zeros(numel(N_Folds), 1);
% 
% for i = 1:numel(N_Folds)
%     cv_mdl = crossval(NaiveBayes_Mdl, "KFold", N_Folds(i));
%     loss = kfoldLoss(cv_mdl);
% 
%     NaiveBayes_Scores(i) = 1 - loss;
% end
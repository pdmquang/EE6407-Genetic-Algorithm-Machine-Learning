figure
% Capped Outliners
X_filled = filloutliers(X, 'center', 'percentiles', [5, 95]);
X_zscore = zscore(X_filled);

boxplot(X_zscore);
title('Compare all features in data train');
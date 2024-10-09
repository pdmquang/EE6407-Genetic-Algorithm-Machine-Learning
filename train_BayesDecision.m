% https://www.mathworks.com/help/stats/discriminant-analysis.html
% https://www.mathworks.com/help/stats/improving-discriminant-analysis-models.html
BayesDecision_Mdl = fitcdiscr(X, label, "ClassNames", {'-1', '1'});

resubLoss(BayesDecision_Mdl)



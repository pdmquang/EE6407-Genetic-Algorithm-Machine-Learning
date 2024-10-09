% https://www.mathworks.com/help/stats/fitcnb.html
NaiveBayes_Mdl = fitcnb(X, label, 'ClassNames', {'-1', '1'});

% class_1 = strcmp(Mdl.ClassNames,'1');
% estimates = Mdl.DistributionParameters{class_1,1}

% disp('job done');
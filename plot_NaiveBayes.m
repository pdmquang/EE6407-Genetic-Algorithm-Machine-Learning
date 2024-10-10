
figure;
hold on;

% Plot decision boundary
gscatter(X_test, X_test, ypreds_Naive);
scatter(class1(:,1), class1(:,2), 'r', 'filled'); % Class 1 points
scatter(class2(:,1), class2(:,2), 'b', 'filled'); % Class 2 points

xlabel('Feature 1');
ylabel('Feature 2');
title('Decision Boundary for Naive Bayes Classifier');
legend('Class Boundary', 'Class 1', 'Class 2');
hold off;




% figure;
% gscatter(X(:,1), X(:,3), label);
% h = gca;
% 
% cxlim = h.XLim;
% cylim = h.YLim;
% 
% hold on
% Params = cell2mat(NaiveBayes_Mdl.DistributionParameters);
% Mu = Params(2*(1:2)-1,1:5); % Extract the means
% Sigma = zeros(5,5,2); % Input data, label(label_size, col, label_size)
% 
% for j = 1:2 % label
%     Sigma(:,:,j) = diag(Params(2*j,:)).^2;
%     xlim = Mu(j,1) + 4*[-1 1]*sqrt(Sigma(1,1,j));
%     ylim = Mu(j,2) + 4*[-1 1]*sqrt(Sigma(2,2,j));
%     f = @(x,y) arrayfun(@(x0,y0) mvnpdf([x0 y0],Mu(j,:),Sigma(:,:,j)),x,y);
%     fcontour(f,[xlim ylim]) % Draw contours for the multivariate normal distributions 
% end
% h.XLim = cxlim;
% h.YLim = cylim;
% title('Naive Bayes Classifier -- NTU Data')
% xlabel('Column 1')
% ylabel('Column 3')
% legend('-1','1')
% hold off
% 

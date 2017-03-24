%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         Confusion tables, ROC and CV                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = dlmread('household.dat','',1,1);
X = data(:,1);
labels = data(:,2);

% Plot the data
% Find indices of class 0
s0 = find(labels==0);
s1 = setdiff((1:20)',s0);
plot(X(s0),0,'ro','MarkerSize',10, 'MarkerFaceColor', 'r'); hold all
plot(X(s1),1,'go','MarkerSize',10,'MarkerFaceColor', 'g')
xlabel('Annual Income')
ylabel('Home ownership')


%%%%%%%%%%%%%%%
% Perform LDA %
%%%%%%%%%%%%%%%
MdlLinear = fitcdiscr(X,labels,'DiscrimType','linear');

% Make a decision
[Yh, score] = predict(MdlLinear,X);

confusionmat(labels,Yh)
plotconfusion(labels', Yh')

% Change decision rule
% Create a vector of zeros
predclass = zeros(size(X,1),1);
predclass(score(:,2)>=0.7)=1;
confusionmat(labels,predclass)
plotconfusion(labels', predclass')

%%%%%%%
% ROC %
%%%%%%%
[xx,yy,T,AUC] = perfcurve(labels,Yh,1);
plotroc(labels',Yh')
title(['ROC for classification by LDA with AUC = ',num2str(AUC)])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         Crossvalidation                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X1 = mvnrnd([0,-1], [1.5,.8;.8,1.5],100);
X2 = mvnrnd([-1,2], [.35,-.2;-.2,.35],100);
X = [X1;X2];
labels = [ones(100,1);2*ones(100,1)];
%Fitting the three classifiers: LDA, QDA and naive Bayes gives
% LDA
MdlLDA = fitcdiscr(X,labels,'DiscrimType','linear');

%QDA
MdlQDA = fitcdiscr(X,labels,'DiscrimType','quadratic');

% Naive Bayes
Mdlnb = fitcnb(X,labels,'Distribution','normal');

%Performing 10 fold cross-validation yields
cvLDA = crossval(MdlLDA,'kfold',10);
cverrorLDA = kfoldLoss(cvLDA)

QDA = crossval(MdlQDA,'kfold',10);
cverrorQDA = kfoldLoss(cvQDA)

cvnb = crossval(Mdlnb,'kfold',10);
cverrornb = kfoldLoss(cvnb)
data = xlsread('banknotes.xlsx');
X = data(:,1:4);
Y = data(:,end);

% Identify a test set to compare several classifiers. 
% Randomly divide 70\% training and 30\% test.
r = randperm(size(X,1));

ntr = ceil(0.75*size(X,1));
ntest = size(X,1)-ntr;
% Make the training and test set
Xtr = X(r(1:ntr),:);
Xtest = X(r(ntr+1:end),:);
Ytr = Y(r(1:ntr));
Ytest = Y(r(ntr+1:end));

% indices of features belonging to class 0
s0 = find(Ytr==0);
% indices of features belonging to class 1. The transpose is to ensure that
% s1 is a column vector (like s0)
s1 = setdiff((1:size(Ytr,1))',s0);

% class 0
HZmvntest(Xtr(s0,:));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                 kNN                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kmax = 25;
cv = zeros(kmax,1);
for k = 1:kmax
    Mdl = fitcknn(Xtr,Ytr,'NumNeighbors',k,'BreakTies','random', 'Distance','euclidean');
    cv(k) = kfoldLoss(crossval(Mdl,'KFold',10)); %10 fold CV
end
% Plot the CV results
plot(1:kmax, cv, 'marker','o','color','r','markersize',6, 'MarkerFaceColor','r')
xlabel('k')
ylabel('Crossvalidation Performance (misclassification rate)')

% Find k with the lowest CV error
[valk,indk] = min(cv);
% Plot the k chosen by CV
line(indk,cv(indk),'color',[.5 .5 .5],'marker','o','linestyle','none','markersize',10, 'MarkerFaceColor','k')

% Fit optimal model
Mdlknn = fitcknn(Xtr,Ytr,'NumNeighbors',indk,'BreakTies','random','Distance','euclidean');

% Make ROC and calculate AUC on TEST
[Yh, score] = predict(Mdlknn,Xtest);

%%%%%%%
% ROC %
%%%%%%%
[xx,yy,T,AUC] = perfcurve(Ytest,Yh,1);
plotroc(Ytest',Yh')
title(['ROC for classification by kNN with AUC = ',num2str(AUC)])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              Naive Bayes with kernel density estimate                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mdlnb = fitcnb(Xtr,Ytr,'Distribution','kernel');
% Make ROC and calculate AUC on TEST
[Yh, score] = predict(Mdlnb,Xtest);

%%%%%%%
% ROC %
%%%%%%%
[xx,yy,T,AUC] = perfcurve(Ytest,Yh,1);
plotroc(Ytest',Yh')
title(['ROC for classification by naive Bayes with AUC = ',num2str(AUC)])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              MC experiment with all classifiers                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
B = 100;
mcrknn = zeros(B,1);
mcrlda = zeros(B,1);
mcrqda = zeros(B,1);
mcrnb = zeros(B,1);
for b = 1:B
    r = randperm(size(X,1));
    
    ntr = ceil(0.75*size(X,1));
    ntest = size(X,1)-ntr;
    % Make the training and test set
    Xtr = X(r(1:ntr),:);
    Xtest = X(r(ntr+1:end),:);
    Ytr = Y(r(1:ntr));
    Ytest = Y(r(ntr+1:end));
    
    %%%%%%%
    % kNN %
    %%%%%%%
    kmax = 25;
    cv = zeros(kmax,1);
    for k = 1:kmax
        Mdl = fitcknn(Xtr,Ytr,'NumNeighbors',k,'BreakTies','random', 'Distance','euclidean');
        cv(k) = kfoldLoss(crossval(Mdl,'KFold',10)); %10 fold CV
    end
    [~,indk] = min(cv);
    Mdlknn = fitcknn(Xtr,Ytr,'NumNeighbors',indk,'BreakTies','random','Distance','euclidean');
    Yh = predict(Mdlknn,Xtest);
    % Misclassification rate
    mcrknn(b,1) = sum(Ytest ~= Yh)/size(Ytest,1);
    
    %%%%%%%
    % LDA %
    %%%%%%%
    Mdllda = fitcdiscr(Xtr,Ytr,'DiscrimType','linear');
    Yh = predict(Mdllda,Xtest);
    % Misclassification rate
    mcrlda(b,1) = sum(Ytest ~= Yh)/size(Ytest,1);
    
    %%%%%%%
    % QDA %
    %%%%%%%
    Mdllda = fitcdiscr(Xtr,Ytr,'DiscrimType','quadratic');
    Yh = predict(Mdllda,Xtest);
    % Misclassification rate
    mcrqda(b,1) = sum(Ytest ~= Yh)/size(Ytest,1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    % naive Bayes with KDE %
    %%%%%%%%%%%%%%%%%%%%%%%%
    Mdlnb = fitcnb(Xtr,Ytr,'Distribution','kernel');
    Yh = predict(Mdlnb,Xtest);
    % Misclassification rate
    mcrnb(b,1) = sum(Ytest ~= Yh)/size(Ytest,1);
end

% Plot all results in a boxplot
legend = {'kNN','LDA','QDA','naive Bayes'};
boxplot([mcrknn mcrlda mcrqda mcrnb],'Labels',legend)
ylabel('Misclassification rate on test data')
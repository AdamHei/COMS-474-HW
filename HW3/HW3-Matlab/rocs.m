data = xlsread('C:/Users/Adam/Desktop/ozon.xlsx');
X = data(:,1:72);
labels = data(:,73);

labels(labels==0) = -1;

svm_model = fitcsvm(X,labels,'KernelFunction','RBF','OptimizeHyperparameters','auto',...
'HyperparameterOptimizationOptions',struct('Optimizer', 'bayesopt','Kfold',10));

svm_model = fitPosterior(svm_model);
[~,score_svm] = resubPredict(svm_model);

[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(labels,score_svm(:,2),1);


model = initlssvm(X,labels,'c',[],[],'RBF_kernel','p');
model = tunelssvm(model,'simplex','crossvalidatelssvm',{10,'misclass'});
model = trainlssvm(model);

roc(model)


plot(Xsvm,Ysvm);
title(['ROC for classification by SVM with AUC = ',num2str(AUCsvm)]);

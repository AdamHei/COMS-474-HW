data = xlsread('C:/Users/Adam/Desktop/ozon.xlsx');
X = data(:,1:72);
labels = data(:,73);

kmax = 100;
mcr_svm = zeros(kmax,1);
mcr_lssvm = zeros(kmax,1);
sample_size = 100;
parfor k = 1:kmax
    r = randperm(size(X,1));
    ntr = ceil(0.75*size(X,1));
    ntest = size(X,1)-ntr;
    
    Xtr = X(r(1:ntr),:);
    Xtest = X(r(ntr+1:end),:);
    Ytr = labels(r(1:ntr));
    Ytest = labels(r(ntr+1:end));    
    
    svm_model = fitcsvm(Xtr,Ytr,'KernelFunction','RBF','OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('Optimizer', 'bayesopt','Kfold',10));
    yhat_svm = predict(svm_model, Xtest);
    mcr_svm(k,1) = sum(Ytest ~= yhat_svm)/size(Ytest,1);
    
    
    model = initlssvm(Xtr,Ytr,'c',[],[],'RBF_kernel','p');
    model = tunelssvm(model,'simplex','crossvalidatelssvm',{10,'misclass'});
    model = trainlssvm(model);
    
    num_rows = size(Xtest,1);
    yhat_lssvm = zeros(num_rows,1);
    for index = 1:size(Xtest,1)
        yhat_lssvm(index) = predict(model, Xtest(index,:), 1);
    end
    
    mcr_lssvm(k,1) = sum(Ytest ~= yhat_lssvm)/size(Ytest,1);
    
    close all
end

legend = {'SVM (w/ RBF)','LSSVM'};
boxplot([mcr_svm mcr_lssvm],'Labels',legend);
ylabel('Misclassification rate on test data');
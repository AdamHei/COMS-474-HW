%% Optimize SVM Classifier

X1 = mvnrnd([0,-1], [1.5,.8;.8,1.5],100);
X2 = mvnrnd([-1,2], [.35,-.2;-.2,.35],100);
X = [X1;X2];
labels = [-ones(100,1);ones(100,1)];
gscatter(X(:,1),X(:,2),labels)
%%
% Find hyperparameters that minimize 10-fold cross-validation loss by
% using automatic hyperparameter optimization.
%
% For reproducibility, set the random seed and use the
% |'expected-improvement-plus'| acquisition function.
Mdl = fitcsvm(X,labels,'KernelFunction','RBF','OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('Optimizer', 'bayesopt','Kfold',10));

%%
% Gridsearch
%%
%Mdl = fitcsvm(X,labels,'KernelFunction','RBF','OptimizeHyperparameters','auto',...
%    'HyperparameterOptimizationOptions',struct('Optimizer', 'gridsearch','NumGridDivisions',10,'Kfold',10))

%Mdl = fitcsvm(X,labels,'KernelFunction','RBF','OptimizeHyperparameters','auto',...
 %   'HyperparameterOptimizationOptions',struct('Optimizer', 'randomsearch','MaxObjectiveEvaluations',200,'Kfold',10))

% Plot classifier
% Create data to evaluate the classifier
delta = 200;
[XX, YY] = meshgrid(linspace(min(X(:,1)),max(X(:,1)),delta),...
    linspace(min(X(:,2)),max(X(:,2)),delta));
Xt = [reshape(XX,numel(XX),1) reshape(YY,numel(YY),1)];
[Yhat,score] = predict(Mdl, Xt);

% Classification plot settings
colormap cool;
map = colormap;

Yhatd = reshape(Yhat,size(XX,1),size(XX,2));
contourf(XX,YY,Yhatd)
hold on

legstr{1} = 'Classifier';
markers = {'*','s','+','o','x','d','v','p','h'};

Y = unique(labels);
nlabels = size(Y,1);
for c = 1:nlabels
    s = find(labels==Y(c));
    plot(X(s,1),X(s,2) ,[markers{1+mod(c-1,9)} 'k']);
    legstr{c+1} = ['class ' num2str(c)];
end
legend(legstr{1:end});
xlabel('X_1')
ylabel('X_2')
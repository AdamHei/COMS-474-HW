%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LS-SVM script for classification %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the fourclass data (binary)
data = xlsread('C:/Users/Adam/Desktop/ozon.xlsx');
X = data(:,1:72);
Y = data(:,73);

% Initialize the model
model = initlssvm(X,Y,'c',[],[],'RBF_kernel','p');

% Tune the hyperparameters with 10-fold CV and minimize the
% misclassification rate
model = tunelssvm(model,'simplex','crossvalidatelssvm',{10,'misclass'});

% Train the model (i.e. solve the linear system)
model = trainlssvm(model);

% If possible, plot results
plotlssvm(model);


%
% MULTICLASS
%

X1 = mvnrnd([0,-1], [1.5,.8;.8,1.5],200);
X2 = mvnrnd([-1,2], [.35,-.2;-.2,.35],200);
X3 = mvnrnd([-3,-6], [.35,-.2;-.2,.35],200);
X = [X1;X2;X3];
labels = [ones(200,1);2*ones(200,1);3*ones(200,1)];

% Initialize the model
model = initlssvm(X,labels,'c',[],[],'RBF_kernel','p');

% Tune the hyperparameters with 10-fold CV and minimize the
% misclassification rate (One vs. One coding)
model = tunelssvm(model,'simplex','crossvalidatelssvm',{10,'misclass'});

% One vs. All coding
%model = tunelssvm(model,'simplex','crossvalidatelssvm',{10,'misclass'},'code_OneVsAll');


% Train the model (i.e. solve the linear system)
model = trainlssvm(model);

% If possible, plot results
plotlssvm(model);


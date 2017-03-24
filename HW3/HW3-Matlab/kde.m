data = dlmread('faithful.dat','',1,1);
X = data(:,2);
% bandwidth selection according to normal reference rule
[f,xi,bw] = ksdensity(X, 'support','positive','function','pdf','kernel','normal');
plot(xi,f)
xlabel('Time between eruptions')
ylabel('density')
function new_duel = bivariate_EI_grid(x, theta, xtrain, ctrain, kernelfun,modeltype, m, kernelname, post)
if m > 2 
error('Bivariate EI only possible with duels, not tournaments')
end
% Bivariate Expected Improvement, as proposed by Nielsen (2015)
x0 = x(:,1);
n= size(x,2);
[~,  g_mu_y, g_sigma2_y, g_Sigma2_y] = prediction_bin(theta, xtrain, ctrain, [x;x0*ones(1,n)], kernelfun, kernelname, modeltype, post, regularization);

id1= find(g_mu_y==max(g_mu_y));
if numel(id1)~=1
    id1 = randsample(id,1);    
end
max_mu = g_mu_y(id1);

DL = g_mu_y - max_mu;
sigma_2_I = g_sigma2_y + g_sigma2_y(id1) - 2*g_Sigma2_y(:,id1);
sigma_I=sqrt(sigma_2_I);
Z=DL./sigma_I;
EI = (DL.*normcdf(Z)+ sigma_I.*normpdf(Z));

id2= find(EI==max(EI));
if numel(id2)~=1
    id2 = randsample(id2,1);    
end
new_duel = [x(:,id1); x(:,id2)];
return



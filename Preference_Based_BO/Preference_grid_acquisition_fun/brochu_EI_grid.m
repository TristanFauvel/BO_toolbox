function new_duel = brochu_EI_grid(x, theta, xtrain, ctrain, kernelfun, xduels, modeltype, m, kernelname, post)

if m > 2 
error('EI only possible with duels, not tournaments')
end

% Bivariate Expected Improvement, as proposed by Brochu (2010)
x0 = x(:,1);
n= size(x,2);
[~,  g_mu_y, g_sigma2_y, g_Sigma2_y] = prediction_bin(theta, xtrain, ctrain, [x;x0*ones(1,n)], kernelfun,kernelname, modeltype, post, regularization);

[max_mu, id1]= max(g_mu_y);

DL = g_mu_y - max_mu;
sigma_y = sqrt(g_sigma2_y);
Z=DL./sigma_y;
EI = (DL.*normcdf(Z)+ sigma_y.*normpdf(Z));%Brochu

id2= find(EI==max(EI));
if numel(id2)~=1
    id2 = randsample(id2,1);    
end
new_duel = [x(:,id1); x(:,id2)];
return



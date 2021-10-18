function new_duel = UCB_LWB_grid(x, theta, xtrain, ctrain, kernelfun,modeltype, m, kernelname)

%(x, theta, xtrain, ctrain, kernelfun, link, xduels,  mu_y_acq, sigma2_y_acq, Sigma2_y_acq, modeltype, maxC, mu_c_acq)
% Bivariate Expected Improvement, as proposed by Brochu (2010)
d = size(x,1);
x0 = x(:,1);
n= size(x,2);
[~,  g_mu_y, g_sigma2_y, ~] = model.prediction(theta, xtrain, ctrain, [x;x0*ones(1,n)], kernelfun, kernelname, modeltype, post, regularization);


kappa = 1;
[f, mu] = ksdensity(g_mu_y,g_mu_y);
likelihood_ratio = 1./f;

ucb_lwb = g_mu_y + kappa*sqrt(g_sigma2_y).*likelihood_ratio;

id1= find(ucb_lwb==max(ucb_lwb));
if numel(id1)~=1
    id1 = randsample(id1,1);    
end


new_duel = NaN(2*d,1);
new_duel(1:d) = x(:,id1);
%% Second member of the duel : 

%Monte-Carlo estimation of the variance of mu_c
ax= ismember(xduels(1:d,:)', new_duel(1:d)', 'rows')';
my_s = mu_y_acq(ax);
sy_2_s = sigma2_y_acq(ax);
Sy_s = Sigma2_y_acq(ax,ax);
mu_c_s=  mu_c_acq(ax);

%samples =mvnrnd(my_s, diag(sy_2_s), 100000);%sample form the p(f|D)

samples =mvnrnd(my_s, nearestSPD(Sy_s), 100000);%sample form the p(f|D)
% V = mean(link(samples).^2,1)' - link(mu_y./sqrt(1+sigma2_y)); % In case were link = sigmoid, replace with : link(my_s./sqrt(1+sy_2_s*pi/8)).^2;
V = mean(link(samples).^2,1)' - mu_c_s.^2; 

maxid= find(V==max(V));
if numel(maxid)~=1
    maxid = randsample(maxid,1);    
end
new_duel(d+1:end,1) = x(:,maxid);

return

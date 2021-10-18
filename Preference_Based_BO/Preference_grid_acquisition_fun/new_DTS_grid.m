function new_duel = new_DTS_grid(x, theta, xtrain, ctrain, kernelfun,modeltype, m,  kernelname)

d = size(x,1);
[~,  mu_y, ~, Sigma2_y] = model.prediction(theta, xtrain, ctrain, xtrain, kernelfun, kernelname,modeltype, post, regularization);
fsamples = mvnrnd(mu_y,  nearestSPD(Sigma2_y))'; %sample from the posterior at training points
Sigma2 = exp(-15); % To regularize

sample_g= sample_value_GP(x, theta, xtrain, fsamples, Sigma2,  kernelname);

idxmax= find(sample_g==max(sample_g));
if numel(idxmax)~=1
    idxmax = randsample(idxmax,1);    
end

new_duel = NaN(2*d,1);
new_duel(1:d) = x(:,idxmax);

%Monte-Carlo estimation of the variance of mu_c
[mu_c_s,  my_s, ~, Sy_s ] = model.prediction(theta, xtrain, ctrain, [x(:,idxmax).*ones(d,size(x,2)); x], kernelfun,kernelname, modeltype, post, regularization);


samples =mvnrnd(my_s, nearestSPD(Sy_s), 1000);%sample form the p(f|D)
% V = mean(link(samples).^2,1)' - link(mu_y./sqrt(1+sigma2_y)); % In case were link = sigmoid, replace with : link(my_s./sqrt(1+sy_2_s*pi/8)).^2;
V = mean(normcdf(samples).^2,1)' - mu_c_s.^2; 

maxid= find(V==max(V));
if numel(maxid)~=1
    maxid = randsample(maxid,1);    
end
new_duel(d+1:end,1) = x(:,maxid);

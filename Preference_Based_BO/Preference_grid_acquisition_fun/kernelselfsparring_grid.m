function new_duel = kernelselfsparring_grid(x, theta, xtrain, ctrain,model,m,  kernelname, post)

[mu_c,  mu_y, sigma2_y, Sigma2_y] = prediction_bin(theta, xtrain, ctrain, xtrain, kernelfun, kernelname, modeltype, post, regularization);

nsamples=m;
gmax=zeros(1, nsamples);
idx =zeros(1, nsamples);
k=0;
Sigma2 = exp(-15); % To regularize

while k < nsamples || numel(unique(idx))<m%sample g* from p(g*|D)
    k=k+1;
    fsamples = mvnrnd(mu_y,  nearestSPD(Sigma2_y))'; %sample from the posterior at training points
    sample_g = sample_value_GP(x, theta, xtrain, fsamples, Sigma2,  kernelname, 'post', post);
    id= find(sample_g==max(sample_g));
    if numel(id)~=1
        id = randsample(id,1);    
    end
    idx(k)=id;
end
idx = unique(idx);
new_duel = x(:,idx); 
new_duel = new_duel(:);


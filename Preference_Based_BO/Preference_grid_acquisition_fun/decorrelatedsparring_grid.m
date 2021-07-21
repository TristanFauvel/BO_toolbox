function new_tour = decorrelatedsparring_grid(x, theta, xtrain, ctrain, kernelfun,modeltype, m, kernelname, post)

[~,  mu_y, ~, Sigma2_y] = prediction_bin_preference(theta, xtrain, ctrain, xtrain, kernelfun, kernelname, 'modeltype', modeltype, 'post', post);
Sigma2 = exp(-15); % To regularize

nsamples=6*m;
idx =zeros(1, nsamples);
k=0;
while k < nsamples || numel(unique(idx))<m%sample g* from p(g*|D)
    k=k+1;
    fsamples = mvnrnd(mu_y,  nearestSPD(Sigma2_y))'; %sample from the posterior at training points
    sample_g = sample_value_GP(x, theta, xtrain, fsamples, Sigma2,  kernelname);
    id= find(sample_g==max(sample_g));
    if numel(id)~=1
        id = randsample(id,1);    
    end
    idx(k)=id;
    if k>nsamples
        disp('Convergence')
    end
end

idx = unique(idx);
x0 = x(:,1);
[~,  ~, sigma2_y, Sigma2_y]  =  prediction_bin_preference(theta, xtrain, ctrain, [x(:,idx);x0*ones(1,numel(idx))], kernelfun, kernelname, 'modeltype', modeltype, 'post', post); %kernelfun(theta, [x(:,idx);x0*ones(1,numel(idx))], [x(:,idx);x0*ones(1,numel(idx))]);

%% Compute the total correlation for any group of m inputs
comb = nchoosek(1:numel(idx),m)'; %compute all possible groups of m
sigma2 = sigma2_y(comb); %compute the variance of all input
detSigma = NaN(1,size(comb,2));
for i = 1:size(comb,2)
    detSigma(i) = det(Sigma2_y(comb(:,i),comb(:,i)));
end

C =  sum(log(sqrt(2*pi*exp(1))*sqrt(sigma2)),1) -0.5*log((2*pi)^m*detSigma); %compute the total correlation

[a,b] = find(C == min(C));
if numel(b)~=1
    b= randsample(b,1);
end
new_tour = x(:,idx(comb(:,b)));
new_tour = new_tour(:);



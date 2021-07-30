function  [new_x,  new_x_norm,idx, L] = TME_sampling(x, theta, xtrain_norm, ytrain,meanfun,kernelfun, lb, ub, post)

xog = x;
ngrid = 300;
if size(x,2)>ngrid
    keep = randsample(size(x,2), ngrid);
    x = x(:,keep);
end

xnorm = (x - lb)./(ub-lb);

% gaussian_entropy = @(Sigma) 0.5+log(det(Sigma)) + 0.5*size(Sigma,1)*log(2*pi*exp(1));
gaussian_entropy = @(sigma2) real(0.5*log(2*pi*exp(1)*sigma2));
regularization = 'nugget';
[mu_y, sigma2_y] =  prediction(theta, xtrain_norm, ytrain, xnorm, kernelfun, meanfun, post, regularization);
H1 = sum(gaussian_entropy(sigma2_y));
n= size(x,2);

%% Expected total marginal entropy after observing y
H2 = zeros(n,1);

for i =1:n      
   [~,sigma2_y] = prediction(theta, [xtrain_norm, xnorm(:,i)], [ytrain, mu_y(i)], xnorm, kernelfun, meanfun, post, regularization); %this computation is based on the fact that the posterior variance does not depend on the observed value.
  
    H2(i) = sum(gaussian_entropy(sigma2_y));
end
L = H1-H2;

idx= find(L==max(L));
if numel(idx)~=1
    idx = randsample(idx,1);
end

new_x = x(:,idx);
new_x_norm = xnorm(:,idx);

idx = find(ismember(xog',new_x', 'rows'));

return


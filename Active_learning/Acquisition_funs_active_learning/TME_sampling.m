function  [new_x,  new_x_norm,idx, L] = TME_sampling(x, theta, xtrain_norm, ytrain,meanfun,kernelfun, lb, ub, post)

xnorm = (x - lb)./(ub-lb);

% gaussian_entropy = @(Sigma) 0.5+log(det(Sigma)) + 0.5*size(Sigma,1)*log(2*pi*exp(1));
gaussian_entropy = @(sigma2) real(0.5*log(2*pi*exp(1)*sigma2));
regularization = 'false';
[mu_y, sigma2_y] =  prediction(theta, xtrain_norm, ytrain, xnorm, kernelfun, meanfun, post, regularization);
H1 = sum(gaussian_entropy(sigma2_y));
n= size(x,2);

%% Expected total marginal entropy after observing y
H2 = zeros(n,1);

nsamples= 1000;
for i =1:n

    post =  prediction(theta, [xtrain_norm, x(:,i)], [], [], kernelfun, meanfun, [], regularization);
    
    samples = mu_y(i) + sqrt(sigma2_y(i))*randn(1,nsamples);

    sigma2_y = zeros(n,nsamples);
    for j = 1:nsamples
        [~,sigma2_y(:, j)] = prediction(theta, [xtrain_norm, x(:,i)], [ytrain, samples(j)], x, kernelfun, meanfun, post, regularization);
    end    
    
    H2(i) = sum(sigma2_y(:))/nsamples;
end
L = H1-H2;

[a,idx] = max(L);
new_x = x(:,idx);
new_x_norm = xnorm(:,idx);

return


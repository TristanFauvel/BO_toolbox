function  [new_x, new_x_norm, idx, L] = TME_sampling_binary(x, theta, xtrain_norm, ctrain, kernelfun, modeltype, lb, ub, post)

xnorm = (x - lb)./(ub-lb);

nx = size(x,2);

% gaussian_entropy = @(Sigma) 0.5+log(det(Sigma)) + 0.5*size(Sigma,1)*log(2*pi*exp(1));
gaussian_entropy = @(sigma2) 0.5*log(2*pi*exp(1)*sigma2);

[mu_c,  mu_y, sigma2_y] =  prediction_bin(theta, xtrain_norm, ctrain, xnorm, kernelfun, 'modeltype', modeltype,'regularization', 'false', 'post', post);

%% Total marginal entropy
H1 = sum(gaussian_entropy(sigma2_y));

%% Expected total marginal entropy after observing y
c0 = [ctrain,0];
c1 = [ctrain,1];
H2 = zeros(size(x,2),1);
for i =1:size(x,2)
    %case y = 0
    [~, ~, sigma2_y0] =  prediction_bin(theta, [xtrain_norm, x(:,i)], c0, xnorm, kernelfun, 'modeltype', modeltype,'regularization', 'false');
    %case y = 1
    [~,~, sigma2_y1] =  prediction_bin(theta, [xtrain_norm, x(:,i)], c1, xnorm, kernelfun, 'modeltype', modeltype,'regularization', 'false');
    H20 = sum(gaussian_entropy(sigma2_y0));
    H21 = sum(gaussian_entropy(sigma2_y1));
    H2(i) = mu_c(i)*H21 + (1-mu_c(i))*H20;
end
L = H1-H2;

[a,idx] = max(L);
new_x = x(:,idx);
new_x_norm = xnorm(:,idx);

return


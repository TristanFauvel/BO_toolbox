function [new_x,new_x_norm, idx, L] = BALD_grid(x, theta, xtrain_norm, ctrain, kernelfun, modeltype,lb, ub, post)

xnorm = (x - lb)./(ub-lb);
regularization = 'false';
[mu_c, mu_y, sigma2_y] =   prediction_bin(theta, xtrain_norm, ctrain, xnorm, kernelfun, modeltype, post, regularization);

C = sqrt(pi*log(2)/2);

h = @(p) -p.*log(p+eps) - (1-p).*log(1-p+eps);

L = h(mu_c) - 0.6932*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);

idx= find(L==max(L));
if numel(idx)~=1
    idx = randsample(idx,1);
end
nd = size(x,1);
if nd==1
    new_x = x(idx);
else
    new_x = x(:,idx);
end
new_x_norm = (new_x - lb)./(ub - lb);

return
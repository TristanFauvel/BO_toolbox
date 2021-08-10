function [new_x,new_x_norm, idx, L] = BALD_grid(x, theta, xtrain_norm, ctrain, kernelfun, modeltype,lb, ub, post)
xog = x;
ngrid = 300;
if size(x,2)>ngrid
    keep = randsample(size(x,2), ngrid);
    x = x(:,keep);
end

xnorm = (x - lb)./(ub-lb);
regularization = ' nugget';
[mu_c, mu_y, sigma2_y] =   prediction_bin(theta, xtrain_norm, ctrain, xnorm, kernelfun, modeltype, post, regularization);
h = @(p) -p.*log(p+eps) - (1-p).*log(1-p+eps);

if strcmp(modeltype, 'exp_prop')
C = sqrt(pi*log(2)/2);
L = h(mu_c) - 0.6932*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
end

idx= find(L==max(L));
if numel(idx)~=1
    idx = randsample(idx,1);
end

new_x = x(:,idx);

new_x_norm = (new_x - lb)./(ub - lb);
idx = find(ismember(xog',new_x', 'rows'));

return
function  [new_x, new_x_norm, idx, L] = maxvar_binary_grid(x, theta, xtrain_norm, ctrain, kernelfun, modeltype, lb, ub, post)
xog = x;
ngrid = 300;
if size(x,2)>ngrid
    keep = randsample(size(x,2), ngrid);
    x = x(:,keep);
end


xnorm = (x - lb)./(ub-lb);

regularization = 'false';
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] =  prediction_bin(theta, xtrain_norm, ctrain, xnorm, kernelfun, modeltype, post, regularization);
L = var_muc;
idx= find(L==max(L));
if numel(idx)~=1
    idx = randsample(idx,1);
end
new_x = x(:,idx);
new_x_norm = xnorm(:,idx);

idx = find(ismember(xog',new_x', 'rows'));

return


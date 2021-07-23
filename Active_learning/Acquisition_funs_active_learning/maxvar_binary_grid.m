function  [new_x, new_x_norm, idx, L] = maxvar_binary_grid(x, theta, xtrain_norm, ctrain, kernelfun, modeltype, lb, ub, post)

xnorm = (x - lb)./(ub-lb);


[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] =  prediction_bin(theta, xtrain_norm, ctrain, xnorm, kernelfun, 'modeltype', modeltype, 'post', post, 'regularization', 'false');

L = var_muc;
[a,idx] = max(L);
new_x = x(:,idx);
new_x_norm = xnorm(:,idx);

return


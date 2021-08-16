function [new_x,new_x_norm] = UCB_binary(theta, xtrain_norm, ctrain, model, post, approximation)

options.method = 'lbfgs';

ncandidates= 5;
init_guess = [];
new_x_norm = multistart_minConf(@(x)ucb(theta, xtrain_norm, x, ctrain, model, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);

new_x = new_x_norm.*(model.ub-model.lb) + model.lb;

end


function [ucb_val, ducb_dx]= ucb(theta, xtrain_norm, x, ctrain, model, post)
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc, dvar_muc_dx] =  prediction_bin(theta, xtrain_norm, ctrain, x, model, post);
sigma_c = sqrt(var_muc);
dsigma_c_dx = dvar_muc_dx./(2*sigma_c);
e = norminv(0.975);
ucb_val = mu_c + e*sigma_c;
ucb_val = -ucb_val;
ducb_dx = -(dmuc_dx(:) + e*dsigma_c_dx(:));
end


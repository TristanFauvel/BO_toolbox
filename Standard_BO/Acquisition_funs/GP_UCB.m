function [new_x, new_x_norm] = GP_UCB(theta, xtrain_norm, ytrain, meanfun, kernelfun, kernelname, max_x, min_x, lb_norm, ub_norm, kernel_approx);        
options.verbose = 1;
ncandidates = 10;

delta = 0.1;
[D,t] = size(xtrain_norm);
e= sqrt(2*log(t.^(0.5*D+2)*pi^2/(3*delta)));
new_x_norm = multistart_minConf(@(x) UCB(theta, xtrain_norm, ytrain, x, kernelfun, meanfun,e),  lb_norm, ub_norm, ncandidates,[], options);
new_x = new_x_norm.*(max_x-min_x) + min_x;
end

function [ucb_val, ducb_dx]= UCB(theta, xtrain_norm, ytrain, x, kernelfun, meanfun,e)
[mu_y, sigma2_y,dmu_dx, dsigma2_dx] =  prediction(theta, xtrain_norm, ytrain, x, kernelfun, meanfun);
sigma_y = sqrt(sigma2_y);
dsigma_y_dx = dsigma2_dx./(2*sigma_y);

ucb_val = mu_y + e*sigma_y;
ucb_val = -ucb_val;
ducb_dx = -(dmu_dx + e*dsigma_y_dx);
end
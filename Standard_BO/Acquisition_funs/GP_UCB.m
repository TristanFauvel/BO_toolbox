function [new_x, new_x_norm] = GP_UCB(theta, xtrain_norm, ytrain, model, post, approximation)   
options.verbose = 1;
ncandidates = 10;

delta = 0.1;
[D,t] = size(xtrain_norm);
e= sqrt(2*log(t.^(0.5*D+2)*pi^2/(3*delta)));
new_x_norm = multistart_minConf(@(x) UCB(theta, xtrain_norm, ytrain, x,model,e, post), model.lb_norm, model.ub_norm, ncandidates,[], options);
new_x = new_x_norm.*(model.ub-model.lb) + model.lb;
end

function [ucb_val, ducb_dx]= UCB(theta, xtrain_norm, ytrain, x, model,e, post)
[mu_y, sigma2_y,dmu_dx, dsigma2_dx] =  prediction(theta, xtrain_norm, ytrain, x, model, post);
sigma_y = sqrt(sigma2_y);
dsigma_y_dx = dsigma2_dx./(2*sigma_y);

ucb_val = mu_y + e*sigma_y;
ucb_val = -ucb_val;
ducb_dx = -(dmu_dx + e*dsigma_y_dx);
end
function  [new_x, new_x_norm] = Dueling_UCB(theta, xtrain_norm, ctrain, model, post, approximation, optim)
% Dueling UCB, (Benavoli 2020)

%% Find the maximum of the value function
options.method = 'lbfgs';

ncandidates = optim.AF_ncandidates;
init_guess = [];

if ~isnan(model.xbest_norm)
    x_duel1_norm = model.xbest_norm;
else
    x_duel1_norm =  model.maxmean(theta, xtrain_norm, ctrain, post);
end

x_duel2_norm = optimize_AF(@(x)dUCB(theta, xtrain_norm, x, ctrain, x_duel1_norm, model, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);

new_x_norm = [x_duel1_norm;x_duel2_norm];
new_x = new_x_norm.*([model.ub;model.ub] - [model.lb; model.lb])+[model.lb; model.lb];

end

function [ucb_val, ducb_dx]= dUCB(theta, xtrain_norm, x, ctrain, x_duel1, model, post)
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx] =  model.prediction(theta, xtrain_norm, ctrain, [x; x_duel1], post);
sigma_y = sqrt(sigma2_y);
dsigma_y_dx = dsigma2y_dx./(2*sigma_y);
D = model.D;
e = norminv(0.975);
ucb_val = mu_y + e*sigma_y;
ducb_dx = (dmuy_dx(1:D) + e*dsigma_y_dx(1:D));
end
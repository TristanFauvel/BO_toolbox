function  [new_x, new_x_norm] = Dueling_UCB_Phi(theta, xtrain_norm, ctrain, model, post, approximation, optim)

options.method = 'lbfgs';
options.verbose = 1;
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

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx,~,~,~, var_muc, dvar_muc_dx] =  model.prediction(theta, xtrain_norm, ctrain, [x;x_duel1], post);
D = model.D;
e = norminv(0.975);

dvar_muc_dx = dvar_muc_dx(1:D)';

ucb_val = mu_c + e*sqrt(var_muc);

dsigma_c_dx = dvar_muc_dx(1:D)./(2*var_muc);

ducb_dx = (dmuc_dx(1:D) + e*dsigma_c_dx);
end
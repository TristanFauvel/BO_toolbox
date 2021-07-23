function [x_duel1, x_duel2,new_duel] = pKG(theta, xtrain_norm, ctrain, kernelfun, base_kernelfun, modeltype, max_x, min_x, lb_norm, ub_norm, condition, post, ~)

% Preference Knowledge Gradient
options.method = 'lbfgs';
options.verbose = 1;
d = size(xtrain_norm,1)/2;
ncandidates =5;
init_guess = [];

[~, gm] = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, kernelfun, condition.x0,modeltype, post), lb_norm, ub_norm, ncandidates,init_guess, options);

g = @(xduel) KG(xduel, theta, xtrain_norm, ctrain, condition, modeltype, post, kernelfun, options, lb_norm, ub_norm, ncandidates, init_guess, gm);

maxiter = 50;
nopt = 20;
kernelfun = @Matern52_kernelfun;
meanfun = @constant_mean;
hyp.cov = [log(1/10),0];
hyp.mean = 0;
ninit = 20;
acquisition_fun = @Thompson_sampling;
ncov_hyp = numel(hyp.cov);
nmean_hyp = numel(hyp.mean);
theta_lb = -5*ones(ncov_hyp  + nmean_hyp ,1);
theta_lb(end) = 0;
theta_ub = 5*ones(ncov_hyp  + nmean_hyp ,1);
theta_ub(end) = 0;
new_duel_norm = BO_loop(g, maxiter, nopt, kernelfun, meanfun, hyp, acquisition_fun, ninit, [ub_norm; ub_norm], [lb_norm;lb_norm], theta_lb, theta_ub, 0);
new_duel = new_duel_norm.*(max_x -min_x) + min_x;
x_duel1 = new_duel(1:d,:);
x_duel2 = new_duel(d+1:end,:);

end

function U = KG(xduel, theta, xtrain_norm, ctrain, condition, modeltype, post, kernelfun, options, lb_norm, ub_norm, ncandidates, init_guess, gm)
mu_c = prediction_bin_preference(theta, xtrain_norm, ctrain, xduel, kernelfun, 'modeltype', modeltype, 'post', post,'regularization', 'false');
[~,~,~,~,~,~,~,~,~,~,post]= prediction_bin_preference(theta,[xtrain_norm, xduel], [ctrain, 1],xduel, kernelfun, 'modeltype', modeltype,'regularization', 'false');
[~, maxg1] = multistart_minConf(@(x)to_maximize_value_function(theta, [xtrain_norm, xduel], [ctrain, 1], x, kernelfun, condition.x0, modeltype, post), lb_norm, ub_norm, ncandidates,init_guess, options);
[~,~,~,~,~,~,~,~,~,~,post]= prediction_bin_preference(theta,[xtrain_norm, xduel], [ctrain, 0],xduel, kernelfun, 'modeltype', modeltype,'regularization', 'false');
[~, maxg0] = multistart_minConf(@(x)to_maximize_value_function(theta, [xtrain_norm, xduel], [ctrain, 0], x, kernelfun, condition.x0, modeltype, post), lb_norm, ub_norm, ncandidates,init_guess, options);
U = mu_c.*(maxg1-gm)+(1-mu_c).*(maxg0-gm);
end
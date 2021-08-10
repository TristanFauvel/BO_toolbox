function [new_x, new_x_norm] = Variance_gradient(theta, xtrain_norm, ctrain, kernelfun,modeltype, max_x, min_x, lb_norm, ub_norm, post)
if ~strcmp(modeltype, 'laplace')
    error('This acquisition function is only implemented with Laplace approximation')
end
ncandidates = 10;
init_guess = [];
options.method = 'lbfgs';
options.verbose = 1;
new_x_norm = multistart_minConf(@(x)adaptive_sampling_binary(x, theta, xtrain_norm, ctrain, kernelfun, modeltype, post), lb_norm, ub_norm, ncandidates,init_guess, options);
new_x = new_x_norm.*(max_x-min_x) + min_x;
end

function [vargrad_x, dvargrad_x_dx] = vargrad(theta, xtrain_norm, ctrain, x_duel1, x, kernelfun, modeltype, post, regularization)
[mu_c,  ~, ~, ~, dmuc_dx,~,~,~, var_muc, dvar_muc_dx] =  prediction_bin(theta, xtrain_norm, ctrain, x, kernelfun, modeltype, post, regularization);

var_muc = -var_muc;
dvar_muc_dx = -dvar_muc_dx;

c0 = [ctrain, 0];
c1 = [ctrain,1];

[~,  ~, ~, ~, ~,~,~,~, var_muc0, dvar_muc0_dx, post0] =  prediction_bin(theta, [xtrain_norm,x], c0, x, kernelfun, modeltype, post, regularization);
[~,  ~, ~, ~, ~,~,~,~, var_muc1, dvar_muc1_dx, post1] =  prediction_bin(theta, [xtrain_norm,x], c1, x, kernelfun, modeltype, post, regularization);

U = var_muc -  mu_c.*var_muc1 + (1-mu_c).*var_muc0

end


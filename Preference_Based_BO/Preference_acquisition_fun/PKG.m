function [new_x, new_x_norm] = PKG(theta, xtrain_norm, ctrain, kernelfun, base_kernelfun, modeltype, max_x, min_x, lb_norm, ub_norm, condition, post, kernel_approx)

if ~strcmp(modeltype, 'laplace')
    error('This acquisition function is only implemented with Laplace approximation')
end
init_guess = [];
options.method = 'lbfgs';
options.verbose = 1;
ncandidates = 5;
[xbest, ybest] = multistart_minConf(@(x)to_maximize_value_GP(theta, xtrain_norm, ctrain, x, kernelfun, condition.x0, modeltype, post), lb_norm, ub_norm, ncandidates, init_guess, options);
ybest = -ybest;

c0 = [ctrain, 0];
c1 = [ctrain,1];

new_x_norm  = multistart_minConf(@(x)knowledge_grad(theta, xtrain_norm, ctrain, x, kernelfun, modeltype, post, c0, c1, xbest, ybest,lb_norm, ub_norm,  condition.x0), lb_norm, ub_norm, ncandidates, init_guess, options);

new_x = new_x_norm.*(max_x-min_x) + min_x;

end
function  [new_x, new_x_norm] = PKG(theta, xtrain_norm, ctrain, model, post, approximation)

if ~strcmp(model.modeltype, 'laplace')
    error('This acquisition function is only implemented with Laplace approximation')
end
init_guess = [];
options.method = 'lbfgs';
options.verbose = 1;
ncandidates =model.ncandidates;
lb_norm = [model.lb_norm; model.lb_norm];
ub_norm = [model.ub_norm; model.ub_norm];

[x_duel1_norm, ybest] = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);
ybest = -ybest;

c0 = [ctrain, 0];
c1 = [ctrain,1];

x_duel2_norm  = multistart_minConf(@(x)knowledge_grad(theta, xtrain_norm, ctrain, x,model, post, c0, c1, x_duel1_norm, ybest,model.lb_norm, model.ub_norm), lb_norm, ub_norm, ncandidates, init_guess, options);

new_x_norm = [x_duel1_norm;x_duel2_norm];
new_x = new_x_norm.*([model.ub;model.ub] - [model.lb; model.lb])+[model.lb; model.lb];
end
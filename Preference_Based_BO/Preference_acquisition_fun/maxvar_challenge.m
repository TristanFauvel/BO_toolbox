function [x_duel1, x_duel2,new_duel] = maxvar_challenge(theta, xtrain_norm, ctrain, model, post, approximation)

options.method = 'lbfgs';
options.verbose = 1;
D = model.D;
ncandidates =model.ncandidates;
init_guess = [];

if ~isnan(post.x_best_norm)
    x_best_norm = post.x_best_norm;
else
    x_best_norm = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);
end

x_duel1 =  x_best_norm.*(model.ub(1:D)-model.lb(1:D)) + model.lb(1:D);

new = multistart_minConf(@(x)pref_var(theta, xtrain_norm, ctrain, x_best_norm , x, model, post), model.lb_norm, model.ub_norm, ncandidates,init_guess, options);
x_duel2 =new.*(model.ub(1:D)-model.lb(1:D)) + model.lb(1:D);

new_duel= [x_duel1; x_duel2];
end

function [var_muc, dvar_muc_dx] = pref_var(theta, xtrain_norm, ctrain, x_duel1, x, model, post)
[~,~,~,~,~,~,~,~, var_muc, dvar_muc_dx] =  model.prediction(theta, xtrain_norm, ctrain, [x;x_duel1], post);
D = size(x,1);
var_muc = -var_muc;
dvar_muc_dx = -dvar_muc_dx(1:D)';
end


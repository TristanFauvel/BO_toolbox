function [x_duel1, x_duel2, new_duel] = DTS(theta, xtrain_norm, ctrain, model, post, approximation)

D= model.D;
init_guess = [];

[x_duel1_norm, x_duel1] = sample_max_preference_GP(approximation, xtrain_norm, ctrain, theta, model, post);

options.method = 'lbfgs';
options.verbose = 1;
ncandidates =model.ncandidates;
 new = multistart_minConf(@(x)pref_var(theta, xtrain_norm, ctrain, x_duel1_norm , x, model, post), model.lb_norm, model.ub_norm, ncandidates,init_guess, options);

x_duel2 =  new.*(model.ub(1:D)-model.lb(1:D)) + model.lb(1:D);
new_duel = [x_duel1;x_duel2];
end

function [var_muc, dvar_muc_dx] = pref_var(theta, xtrain_norm, ctrain, x_duel1_norm, x, model, post)
[~,~,~,~,~,~,~,~, var_muc, dvar_muc_dx] =  prediction_bin(theta, xtrain_norm, ctrain, [x;x_duel1_norm], model, post);
D = size(x,1);
var_muc = -var_muc;
dvar_muc_dx = -dvar_muc_dx(1:D)';
end

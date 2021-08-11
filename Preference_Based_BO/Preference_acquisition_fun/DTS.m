function [x_duel1, x_duel2, new_duel] = DTS(theta, xtrain_norm, ctrainmodel, post, approximation)

D = size(xtrain_norm,1)/2;
decoupled_bases = 1;

init_guess = [];

[x_duel1_norm, x_duel1] = sample_max_preference_GP(approximation, xtrain_norm, ctrain, theta, model, decoupled_bases, post);

options.method = 'lbfgs';
options.verbose = 1;
ncandidates= 5;
regularization= 'nugget';
new = multistart_minConf(@(x)pref_var(theta, xtrain_norm, ctrain, x_duel1_norm , x, model, post), lb_norm, ub_norm, ncandidates,init_guess, options);

x_duel2 = new.*(max_x(1:D)-min_x(1:D)) + min_x(1:D);
new_duel = [x_duel1;x_duel2];
end

function [var_muc, dvar_muc_dx] = pref_var(theta, xtrain_norm, ctrain, x_duel1_norm, x, model, post)
[~,~,~,~,~,~,~,~, var_muc, dvar_muc_dx] =  prediction_bin(theta, xtrain_norm, ctrain, [x;x_duel1_norm], model, post);
D = size(x,1);
var_muc = -var_muc;
dvar_muc_dx = -dvar_muc_dx(1:D)';
end

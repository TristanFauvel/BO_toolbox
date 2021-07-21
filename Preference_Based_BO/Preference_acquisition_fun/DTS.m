function [x_duel1, x_duel2, new_duel] = DTS(theta, xtrain_norm, ctrain, kernelfun, base_kernelfun, modeltype, max_x, min_x, lb_norm, ub_norm, condition, post, kernel_approx)

D = size(xtrain_norm,1)/2;
decoupled_bases = 1;

init_guess = [];

[x_duel1_norm, x_duel1] = sample_max_preference_GP(kernel_approx, xtrain_norm, ctrain, theta,kernelfun, decoupled_bases, modeltype, base_kernelfun, post, condition, max_x, min_x, lb_norm, ub_norm);

options.method = 'lbfgs';
options.verbose = 1;
ncandidates= 5;
new = multistart_minConf(@(x)pref_var(theta, xtrain_norm, ctrain, x_duel1_norm , x, kernelfun, modeltype, post), lb_norm, ub_norm, ncandidates,init_guess, options);

x_duel2 = new.*(max_x(1:D)-min_x(1:D)) + min_x(1:D);
new_duel = [x_duel1;x_duel2];
end

function [var_muc, dvar_muc_dx] = pref_var(theta, xtrain_norm, ctrain, x_duel1_norm, x, kernelfun, modeltype, post)
[~,~,~,~,~,~,~,~, var_muc, dvar_muc_dx] =  prediction_bin_preference(theta, xtrain_norm, ctrain, [x;x_duel1_norm], kernelfun, 'modeltype', modeltype, 'post', post);
D = size(x,1);
var_muc = -var_muc;
dvar_muc_dx = -dvar_muc_dx(1:D)';
end

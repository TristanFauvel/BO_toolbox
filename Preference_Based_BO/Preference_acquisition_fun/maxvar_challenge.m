function [x_duel1, x_duel2,new_duel] = maxvar_challenge(theta, xtrain_norm, ctrain, kernelfun, base_kernelfun, modeltype, max_x, min_x, lb_norm, ub_norm, condition, post, ~)

options.method = 'lbfgs';
options.verbose = 1;
D = size(xtrain_norm,1)/2;
ncandidates =5;
init_guess = [];

x_best_norm = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, kernelfun, condition.x0,modeltype, post), lb_norm, ub_norm, ncandidates,init_guess, options);

x_duel1 =  x_best_norm.*(max_x(1:D)-min_x(1:D)) + min_x(1:D);


new = multistart_minConf(@(x)pref_var(theta, xtrain_norm, ctrain, x_best_norm , x, kernelfun, modeltype, post), lb_norm, ub_norm, ncandidates,init_guess, options);
x_duel2 = new.*(max_x(1:D)-min_x(1:D)) + min_x(1:D);

new_duel= [x_duel1; x_duel2];
end

function [var_muc, dvar_muc_dx] = pref_var(theta, xtrain_norm, ctrain, x_duel1, x, kernelfun, modeltype, post)
[~,~,~,~,~,~,~,~, var_muc, dvar_muc_dx] =  prediction_bin_preference(theta, xtrain_norm, ctrain, [x;x_duel1], kernelfun, 'modeltype', modeltype, 'post', post);
D = size(x,1);
var_muc = -var_muc;
dvar_muc_dx = -dvar_muc_dx(1:D)';
end


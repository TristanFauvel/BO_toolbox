function  [new_x, new_x_norm] = MUC(theta, xtrain_norm, ctrain, model, post, approximation, optim)

options.method = 'lbfgs';
options.verbose = 1;
ncandidates = optim.AF_ncandidates;
init_guess = [];

if ~isnan(model.xbest_norm)
    x_duel1_norm = model.xbest_norm;
else
    x_duel1_norm =  model.maxmean(theta, xtrain_norm, ctrain, post);
end

 
x_duel2_norm= optimize_AF(@(x)pref_var(theta, xtrain_norm, ctrain, x_duel1_norm , x, model, post), model.lb_norm, model.ub_norm, ncandidates,init_guess, options);

new_x_norm = [x_duel1_norm;x_duel2_norm];
new_x = new_x_norm.*([model.ub;model.ub] - [model.lb; model.lb])+[model.lb; model.lb];
end

function [var_muc, dvar_muc_dx] = pref_var(theta, xtrain_norm, ctrain, x_duel1, x, model, post)
[~,~,~,~,~,~,~,~, var_muc, dvar_muc_dx] =  model.prediction(theta, xtrain_norm, ctrain, [x;x_duel1], post);
D = size(x,1);
dvar_muc_dx = dvar_muc_dx(1:D)';
end


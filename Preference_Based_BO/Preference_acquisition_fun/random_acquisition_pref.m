function [new_x, new_x_norm] = random_acquisition_pref(theta, xtrain_norm, ctrain, model, post, approximation, optim)

samples = rand_interval(model.lb_norm, model.ub_norm, 'nsamples', 2);

x_duel1_norm = samples(:,1);
x_duel2_norm = samples(:,2);

new_x_norm = [x_duel1_norm;x_duel2_norm];
new_x = new_x_norm.*([model.ub;model.ub] - [model.lb; model.lb])+[model.lb; model.lb];

 
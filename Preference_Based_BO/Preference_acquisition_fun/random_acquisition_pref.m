function [x_duel1, x_duel2, new_duel] = random_acquisition_pref(theta, xtrain_norm, ctrain, model, post, approximation)

samples = rand_interval(model.lb, model.ub, 'nsamples', 2);

x_duel1 = samples(:,1);
x_duel2 = samples(:,2);
new_duel= [x_duel1; x_duel2];

 
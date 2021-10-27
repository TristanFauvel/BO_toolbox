function  [new_x, new_x_norm] = MaxEntChallenge(theta, xtrain_norm, ctrain, model, post, approximation)
options.method = 'lbfgs';
options.verbose = 1;
D = model.D;
ncandidates =model.ncandidates;
init_guess = [];

if ~isnan(model.xbest_norm)
    x_duel1_norm = model.xbest_norm;
else
    x_duel1_norm =  model.maxmean(theta, xtrain_norm, ctrain, post);
end
 
model2 = model;
model2.lb_norm = [model.lb_norm;x_best_norm];
model2.ub_norm = [model.ub_norm;x_best_norm];

[x_duel2, x_duel2_norm] = active_sampling_binary(theta, xtrain_norm, ctrain, model2, post);

x_duel2_norm = x_duel2_norm(1:D);
 
new_x_norm = [x_duel1_norm;x_duel2_norm];
new_x = new_x_norm.*([model.ub;model.ub] - [model.lb; model.lb])+[model.lb; model.lb];



end


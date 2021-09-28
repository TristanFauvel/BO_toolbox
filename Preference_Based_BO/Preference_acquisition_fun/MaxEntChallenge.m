function [x_duel1, x_duel2,new_duel] = MaxEntChallenge(theta, xtrain_norm, ctrain, model, post, approximation)
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

model2 = model;
model2.lb_norm = [model.lb_norm;x_best_norm];
model2.ub_norm = [model.ub_norm;x_best_norm];

x_duel2 = active_sampling_binary(theta, xtrain_norm, ctrain, model2, post);

 x_duel2 = x_duel2(1:D);
 
new_duel= [x_duel1; x_duel2];
end


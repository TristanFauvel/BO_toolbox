function [new_x, new_x_norm] = BALD(theta, xtrain_norm, ctrain, model, post, ~)
ncandidates = 10;
init_guess = [];
options.method = 'lbfgs';
options.verbose = 1;

if strcmp(model.type, 'preference') && numel(model.ns)>0
        model.lb_norm = [model.lb_norm;model.lb_norm((end-model.D+1):end)];
        model.ub_norm = [model.ub_norm;model.ub_norm((end-model.D+1):end)];
end
new_x_norm = multistart_minConf(@(x)  Entropic_espistemic_uncertainty(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options, 'objective', 'max');
new_x = new_x_norm.*(model.max_x-model.min_x) + model.min_x;
end
 

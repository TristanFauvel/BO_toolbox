function [new_x, new_x_norm] = BALD(theta, xtrain_norm, ctrain, model, post, ~, optim)
ncandidates = optim.AF_ncandidates;
init_guess = [];
options.method = 'lbfgs';
options.verbose = 1;

if strcmp(model.type, 'preference') && model.ns>0
        model.lb_norm = [model.lb_norm;model.lb_norm((end-model.D+1):end)];
        model.ub_norm = [model.ub_norm;model.ub_norm((end-model.D+1):end)];
end
new_x_norm = optimize_AF(@(x)  Entropic_epistemic_uncertainty(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options, 'objective', 'max');
new_x = new_x_norm.*(model.ub - model.lb) + model.lb;
end
 

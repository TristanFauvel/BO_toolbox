function [sample_normalized, sample] = sample_max_binary_GP(approximation, xtrain_norm, ctrain, theta, model, post)
phi = approximation.phi;
dphi_dx = approximation.dphi_dx;

options.method = 'lbfgs';
options.verbose = 1;
ncandidates= 5;

[sample_g, dsample_g_dx] = sample_binary_GP_precomputed_features(phi, dphi_dx, xtrain_norm, ctrain, theta,model, approximation, post);

init_guess = [];

sample_normalized = multistart_minConf(@(x)deriv(x,sample_g, dsample_g_dx), model.lb_norm, model.ub_norm, ncandidates,init_guess, options);
 sample = sample_normalized.*(model.ub-model.lb) + model.lb;
end

function [fx, dfdx] = deriv(x,f,df)
if any(isnan(x))
    warning('x is NaN')
end
% Function that groups f and df to use minFunc
fx = -f(x);
dfdx = - df(x);
end
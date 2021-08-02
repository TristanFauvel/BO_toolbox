function [sample_normalized, sample] = sample_max_preference_GP(kernel_approx, xtrain_norm, ctrain, theta,kernelfun, decoupled_bases, modeltype, base_kernelfun, post, condition, max_x, min_x, lb_norm, ub_norm)

D = size(xtrain_norm,1)/2;
phi_pref = kernel_approx.phi_pref;
dphi_pref_dx = kernel_approx.phi_pref;
phi = kernel_approx.phi;
dphi_dx = kernel_approx.dphi_dx;

options.method = 'lbfgs';
options.verbose = 1;
ncandidates= 5;

[sample_g, dsample_g_dx] = sample_value_GP_precomputed_features(phi, dphi_dx, phi_pref, dphi_pref_dx, xtrain_norm, ctrain, theta,kernelfun, decoupled_bases, modeltype, base_kernelfun, post, condition);

init_guess = [];
new = multistart_minConf(@(x)deriv(x,sample_g, dsample_g_dx), lb_norm, ub_norm, ncandidates,init_guess, options);

%Careful here: the goal is to maximize the value function (the problem
%is a maximization problem): deriv takes the opposite of sample_g
sample = new.*(max_x(1:D)-min_x(1:D)) + min_x(1:D);
sample_normalized= new;
end

function [fx, dfdx] = deriv(x,f,df)
if any(isnan(x))
    warning('x is NaN')
end
% Function that groups f and df to use minFunc
fx = -f(x);
dfdx = - df(x);
end



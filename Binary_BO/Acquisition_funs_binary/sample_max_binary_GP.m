function [sample_normalized, sample] = sample_max_binary_GP(approximation, xtrain_norm, ctrain, theta,kernelfun, decoupled_bases, modeltype,post, max_x, min_x, lb_norm, ub_norm)

D = size(xtrain_norm,1);
phi = approximation.phi;
dphi_dx = approximation.dphi_dx;

options.method = 'lbfgs';
options.verbose = 1;
ncandidates= 5;

[sample_g, dsample_g_dx] = sample_binary_GP_precomputed_features(phi, dphi_dx, xtrain_norm, ctrain, theta,model, decoupled_bases, post);

new= NaN;
init_guess = [];
while isnan(new)
    new = multistart_minConf(@(x)deriv(x,sample_g, dsample_g_dx), lb_norm, ub_norm, ncandidates,init_guess, options);
    if any(isnan(new))
        warning('x is NaN')
    end
end

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
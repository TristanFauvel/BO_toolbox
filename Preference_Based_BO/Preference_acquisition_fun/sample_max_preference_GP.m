function [sample_normalized, sample] = sample_max_preference_GP(approximation, xtrain_norm, ctrain, theta, model, post)

 
options.method = 'lbfgs';
options.verbose = 1;
ncandidates = 10;

[sample_g, dsample_g_dx] = sample_value_GP_precomputed_features(approximation, theta, xtrain_norm, ctrain, post);

init_guess = [];
new = multistart_minConf(@(x)deriv(x,sample_g, dsample_g_dx), model.lb_norm, model.ub_norm, ncandidates,init_guess, options);

%Careful here: the goal is to maximize the value function (the problem
%is a maximization problem): deriv takes the opposite of sample_g
sample = new.*(model.ub -model.lb) + model.lb;
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



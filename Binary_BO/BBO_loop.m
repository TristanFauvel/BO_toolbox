function [xtrain, xtrain_norm, ctrain, score]= BBO_loop(acquisition_fun, nopt, seed, lb, ub, maxiter, theta, g, update_period, modeltype, theta_lb, theta_ub, kernelname, kernelfun, lb_norm, ub_norm, link);

% g : objective function

% maxiter : number of iterations
% nopt : number of time steps before starting using the acquisition
% ninit : number of time steps before starting updating the hyperparameters
% function

D = numel(ub);
lb_norm = zeros(D,1);
ub_norm = ones(D,1);


xtrain = [];
xtrain_norm = [];
ctrain = [];


options_theta.method = 'lbfgs';
options_theta.verbose = 1;

rng(seed)
new_x_norm = rand_interval(lb_norm,ub_norm);
new_x = new_x_norm.*(ub - lb)+lb;
ninit= maxiter + 2;

options.method = 'lbfgs';
ncandidates= 10;
%% Compute the kernel approximation if needed
if strcmp(kernelname, 'Matern52') || strcmp(kernelname, 'Matern32') || strcmp(kernelname, 'ARD')
    approximation_method = 'RRGP';
else
    approximation_method = 'SSGP';
end
nfeatures = 256;
[kernel_approx.phi, kernel_approx.dphi_dx] = sample_features_GP(theta(:), D, kernelname, approximation_method, nfeatures);
regularization = 'nugget';
for i =1:maxiter
    disp(i)
    new_c = g(new_x)>rand;
    xtrain = [xtrain, new_x];
    xtrain_norm = [xtrain_norm, new_x_norm];
    ctrain = [ctrain, new_c];
      
    
    if i > ninit
        %Local optimization of hyperparameters
        if mod(i, update_period) ==0
            init_guess = theta;
            theta = multistart_minConf(@(hyp)minimize_negloglike_bin(hyp, xtrain_norm, ctrain, kernelfun, meanfun, update, post), theta_lb, theta_ub,10, init_guess, options_theta);
        end
    end
    post =  prediction_bin(theta, xtrain_norm, ctrain, [], kernelfun, modeltype, [], regularization);

    if i> nopt
        [new_x, new_x_norm] = acquisition_fun(theta, xtrain_norm, ctrain, kernelfun, modeltype, ub, lb, lb_norm, ub_norm,post, kernel_approx);
    else
        new_x_norm = rand_interval(lb_norm,ub_norm);
        new_x = new_x_norm.*(ub - lb)+lb;
    end
    init_guess = [];
    x_best_norm(:,i) = multistart_minConf(@(x)to_maximize_mean_bin_GP(theta, xtrain_norm, ctrain, x, kernelfun,modeltype, post), lb_norm, ub_norm, ncandidates, init_guess, options);
    x_best(:,i) = x_best_norm(:,i) .*(ub-lb) + lb;
    score(i) = g(x_best(:,i));
end


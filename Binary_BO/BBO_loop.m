function [xtrain, xtrain_norm, ctrain, score,x_best]= BBO_loop(acquisition_fun, nopt, seed, maxiter, theta, g, update_period, model)

% g : objective function
% maxiter : number of iterations
% nopt : number of time steps before starting using the acquisition
% ninit : number of time steps before starting updating the hyperparameters
% function

ub = model.ub;
lb= model.lb;
lb_norm = model.lb_norm;
ub_norm = model.ub_norm;

D = numel(ub);

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
if strcmp(model.kernelname, 'Matern52') || strcmp(model.kernelname, 'Matern32') || strcmp(model.kernelname, 'ARD')
    approximation.method = 'RRGP';
else
    approximation.method = 'SSGP';
end
approximation.decoupled_bases = 1;
approximation.nfeatures = 256;
[approximation.phi, approximation.dphi_dx] = sample_features_GP(theta(:), model, approximation);
x_best_norm = zeros(D, maxiter);
x_best = zeros(D, maxiter);
score = zeros(1,maxiter);


identification = 'mu_c';
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
            [approximation.phi, approximation.dphi_dx] = sample_features_GP(theta(:), model, approximation);
            
        end
    end
    post =  prediction_bin(theta, xtrain_norm, ctrain, [], model, []);
    
    if i> nopt
        [new_x, new_x_norm] = acquisition_fun(theta, xtrain_norm, ctrain,model, post, approximation);
    else
        new_x_norm = rand_interval(lb_norm,ub_norm);
        new_x = new_x_norm.*(ub - lb)+lb;
    end
    init_guess = [];
    
    if strcmp(identification, 'mu_c')
        x_best_norm(:,i) = multistart_minConf(@(x)to_maximize_mu_c_GP(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);
    elseif strcmp(identification, 'mu_g')
        x_best_norm(:,i) = multistart_minConf(@(x)to_maximize_mean_bin_GP(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);
    end
    
    x_best(:,i) = x_best_norm(:,i) .*(ub-lb) + lb;
    score(i) = g(x_best(:,i));
end


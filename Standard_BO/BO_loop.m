function [xtrain, xtrain_norm, ytrain, score, cum_regret]= BO_loop(g, maxiter, nopt, kernelfun, meanfun, theta, acquisition_fun, ninit, max_x, min_x, theta_lb, theta_ub, max_g, kernelname, lb_norm, ub_norm)

% Run a Bayesian optimization loop
% g : objective function
% max_g : maximum of the objective function
% maxiter : number of iterations
% nopt : number of time steps before starting using the acquisition
% ninit : number of time steps before starting updating the hyperparameters
% function

D = numel(max_x);

xtrain = [];
xtrain_norm = [];
ytrain = [];

cum_regret_i =0;
cum_regret=NaN(1, maxiter+1);
cum_regret(1)=0;
score = NaN(1,maxiter);

options_theta.method = 'lbfgs';
options_theta.verbose = 1;
ncov_hyp = numel(theta.cov);
nmean_hyp = numel(theta.mean);
% theta_lb = -8*ones(ncov_hyp  + nmean_hyp ,1);
% theta_lb(end) = 0;
% theta_ub = 10*ones(ncov_hyp  + nmean_hyp ,1);
% theta_ub(end) = 0;

new_x_norm = rand_interval(lb_norm,ub_norm);
new_x = new_x_norm.*(max_x - min_x)+min_x;

if strcmp(kernelname, 'Matern52') || strcmp(kernelname, 'Matern32') %|| strcmp(kernelname, 'ARD')
    approximation_method = 'RRGP';
else
    approximation_method = 'SSGP';
end
nfeatures = 4096;
%kernel_approx.phi : ntest x nfeatures
%kernel_approx.phi_pref : ntest x nfeatures
% kernel_approx.dphi_dx : nfeatures x D
% kernel_approx.dphi_dx : nfeatures x 2D
[kernel_approx.phi, kernel_approx.dphi_dx]= sample_features_GP(theta.cov, D, kernelname, approximation_method, nfeatures);

    
for i =1:maxiter
    i
    new_y = g(new_x);
    xtrain = [xtrain, new_x];
    xtrain_norm = [xtrain_norm, new_x_norm];
    ytrain = [ytrain, new_y];
    
    mu_ytrain =  prediction(theta, xtrain_norm, ytrain, xtrain_norm, kernelfun, meanfun);
    [max_ytrain,b]= max(mu_ytrain);
    
    cum_regret_i  =cum_regret_i + max_g-max_ytrain;
    cum_regret(i+1) = cum_regret_i;       
    score(i) =  g(xtrain(:,b));
    
    if i > ninit
        update = 'cov';       
        init_guess = [theta.cov; theta.mean];
        hyp = multistart_minConf(@(hyp)minimize_negloglike(hyp, xtrain_norm, ytrain, kernelfun, meanfun, ncov_hyp, nmean_hyp, update), theta_lb, theta_ub,10, init_guess, options_theta); 
        theta.cov = hyp(1:ncov_hyp);
        theta.mean = hyp(ncov_hyp+1:ncov_hyp+nmean_hyp);
    end
    if i> nopt              
        [new_x, new_x_norm] = acquisition_fun(theta, xtrain_norm, ytrain, meanfun, kernelfun, kernelname, max_x, min_x, lb_norm, ub_norm, kernel_approx);        
    else
        new_x_norm = rand_interval(lb_norm,ub_norm);
        new_x = new_x_norm.*(max_x - min_x)+min_x;
    end
end
mu_ytrain =  prediction(theta, xtrain_norm, ytrain, xtrain, kernelfun, meanfun);
% [max_ytrain,b]= max(mu_ytrain);
% max_xtrain = xtrain(:,b);


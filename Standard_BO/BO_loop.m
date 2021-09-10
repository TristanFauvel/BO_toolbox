function [xtrain, xtrain_norm, ytrain, score, xbest, cum_regret, theta_evo]= BO_loop(g, maxiter, nopt, model, theta, acquisition_fun, ninit, max_g, seed)
% Run a Bayesian optimization loop
% g : objective function
% max_g : maximum of the objective function
% maxiter : number of iterations
% nopt : number of time steps before starting using the acquisition
% ninit : number of time steps before starting updating the hyperparameters
% function

D = numel(model.ub);

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
rng(seed)


new_x_norm = rand_interval(model.lb_norm,model.ub_norm);
new_x = new_x_norm.*(model.ub - model.lb)+model.lb;

if strcmp(model.kernelname, 'Matern52') || strcmp(model.kernelname, 'Matern32') %|| strcmp(kernelname, 'ARD')
    approximation.method = 'RRGP';
else
    approximation.method = 'SSGP';
end
approximation.nfeatures = 4096;
approximation.decoupled_bases = 1;
%approximation.phi : ntest x nfeatures
%approximation.phi_pref : ntest x nfeatures
% approximation.dphi_dx : nfeatures x D
% approximation.dphi_dx : nfeatures x 2D
[approximation.phi, approximation.dphi_dx]= sample_features_GP(theta.cov, D, model, approximation);
post = [];
theta_evo = zeros(numel(theta.cov), maxiter);
rng(seed)
for i =1:maxiter
    i
    new_y = g(new_x);
    xtrain = [xtrain, new_x];
    xtrain_norm = [xtrain_norm, new_x_norm];
    ytrain = [ytrain, new_y];
    
    mu_ytrain =  prediction(theta, xtrain_norm, ytrain, xtrain_norm, model, post);
    [max_ytrain,b]= max(mu_ytrain);
    
    cum_regret_i  =cum_regret_i + max_g-max_ytrain;
    cum_regret(i+1) = cum_regret_i;       
    score(i) =  g(xtrain(:,b));
    xbest(:,i) = xtrain(:,b);
    if i > ninit
        update = 'cov';       
        init_guess = [theta.cov; theta.mean];
        hyp = multistart_minConf(@(hyp)minimize_negloglike(hyp, xtrain_norm, ytrain, model.kernelfun, model.meanfun, ncov_hyp, nmean_hyp, update), theta_lb, theta_ub,10, init_guess, options_theta); 
        theta.cov = hyp(1:ncov_hyp);
        theta.mean = hyp(ncov_hyp+1:ncov_hyp+nmean_hyp);
        [approximation.phi, approximation.dphi_dx]= sample_features_GP(theta.cov, D, model, approximation);
    end
    if i> nopt              
        [new_x, new_x_norm] = acquisition_fun(theta, xtrain_norm, ytrain, model, post, approximation);        
    else
        new_x_norm = rand_interval(model.lb_norm,model.ub_norm);
        new_x = new_x_norm.*(model.ub - model.lb)+model.lb;
    end
    theta_evo(:, i) = theta.cov;
end

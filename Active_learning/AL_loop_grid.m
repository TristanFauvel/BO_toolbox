function [xtrain, ytrain, cum_regret, score]= AL_loop_grid(x, y, maxiter, nopt, kernelfun, meanfun, theta, acquisition_fun, ninit, theta_lb, theta_ub, lb, ub, seed)

new_i = randsample(size(x,2),1);
new_x = x(:,new_i);
new_x_norm = (new_x - lb)./(ub - lb);

D = size(x,1);

cum_regret=NaN(1, maxiter+1);
cum_regret(1)=0;

rng(seed)

xtrain = NaN(D,maxiter);
xtrain_norm = NaN(D,maxiter);

ytrain = NaN(1, maxiter);
score = NaN(1,maxiter);
regularization = 'nugget';
for i =1:maxiter
    xtrain(:,i) = new_x;
    ytrain(:,i) = y(new_i);
           xtrain_norm(:,i) = new_x_norm;

    [mu_y,sigma2_y] = prediction(theta, xtrain(:,1:i), ytrain(:,1:i), x, kernelfun, meanfun, [], regularization);
  
    
    Err = sigma2_y(:)+(y(:)-mu_y(:)).^2;
    
    score(i) = mean(Err);
    
    cum_regret(i+1) = cum_regret(i)+score(i);
    
    if i > ninit
        update = 'cov';       
        init_guess = [theta.cov; theta.mean];
        hyp = multistart_minConf(@(hyp)minimize_negloglike(hyp, xtrain_norm(:,1:i), ytrain(:,1:i), kernelfun, meanfun, ncov_hyp, nmean_hyp, update), theta_lb, theta_ub,10, init_guess, options_theta); 
        theta.cov = hyp(1:ncov_hyp);
        theta.mean = hyp(ncov_hyp+1:ncov_hyp+nmean_hyp);
    end
    if i> nopt              
        [new_x, new_x_norm, new_i] = acquisition_fun(x, theta,  xtrain_norm(:,1:i), ytrain(:,1:i), meanfun, kernelfun, lb, ub, []);        
    else
        new_x = rand_interval(lb,ub);
        new_x_norm = (new_x - lb)./(ub - lb);
    end
end
return

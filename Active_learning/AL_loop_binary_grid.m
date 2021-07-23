function [xtrain, ctrain, cum_regret, score]= AL_loop_binary_grid(x, y, maxiter, nopt, kernelfun, theta, acquisition_fun, ninit, theta_lb, theta_ub, lb, ub, seed)

new_i = randsample(size(x,2),1);
new_x = x(:,new_i);
new_x_norm = (new_x - lb)./(ub-lb);
D = size(x,1);

cum_regret=NaN(1, maxiter+1);
cum_regret(1)=0;

rng(seed)

xtrain = NaN(D,maxiter);
xtrain_norm = NaN(D,maxiter);

ctrain = NaN(1, maxiter);
score = NaN(1,maxiter);
modeltype = 'exp_prop';

if strcmp(func2str(acquisition_fun), 'random')
    nopt = maxiter +1;
end

for i =1:maxiter
    disp(i)
    xtrain(:,i) = new_x;
    ctrain(:,i) = normcdf(y(new_i))>rand;
    xtrain_norm(:,i) = new_x_norm;
    [mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc, dvar_muc_dx, post] = prediction_bin(theta, xtrain(:,1:i), ctrain(:,1:i), x, kernelfun, 'modeltype', modeltype,'regularization', 'false');
  
    score(i) = sqrt(mse(mu_y',normcdf(y)));
    cum_regret(i+1) = cum_regret(i)+score(i);
    
    if i > ninit
        init_guess = theta;
        theta = multistart_minConf(@(hyp)negloglike_bin(hyp, xtrain_norm(:,1:i), ctrain(:,1:i), kernelfun, 'modeltype', modeltype, 'post', post), theta_lb, theta_ub,10, init_guess, options_theta); 
    end
    if i> nopt              
        [new_x, new_x_norm, new_i] = acquisition_fun(x, theta,  xtrain_norm(:,1:i), ctrain(:,1:i), kernelfun, modeltype, lb, ub, post);        
    else
        new_x = rand_interval(lb,ub);
        new_x_norm = (new_x - lb)./(ub - lb);
    end
end
return

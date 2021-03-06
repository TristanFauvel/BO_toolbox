function [xtrain, ctrain, cum_regret, score_y, score_c]= AL_loop_binary_grid(x, y, maxiter, nopt, model, theta, acquisition_fun, ninit, seed)

rng(seed)

new_i = randsample(size(x,2),1);
new_x = x(:,new_i);
new_x_norm = (new_x - model.lb)./(model.ub - model.lb);
[D,N] = size(x);

cum_regret=NaN(1, maxiter+1);
cum_regret(1)=0;

 
y = mvnrnd(zeros(size(y)),model.kernelfun(theta.cov, x,x, true, model.regularization))'; %%%%%%%%%%%%%

xtrain = NaN(D,maxiter);
xtrain_norm = NaN(D,maxiter);

ctrain = NaN(1, maxiter);
score_c = NaN(1,maxiter);
score_y = NaN(1,maxiter);

modeltype = 'exp_prop';
link = @normcdf;
if strcmp(func2str(acquisition_fun), 'random')
    nopt = maxiter +1;
end
options_theta.verbose = 1;
options_theta.method = 'lbfgs';
post = [];
for i =1:maxiter
    disp(i)
    xtrain(:,i) = new_x;
    ctrain(:,i) = link(y(new_i))>rand;
    xtrain_norm(:,i) = new_x_norm;
    post = model.prediction(theta, xtrain_norm(:,1:i), ctrain(:,1:i), [], post);

    [mu_c, mu_y, sigma2_y, ~, ~, ~, ~, ~, var_muc] = model.prediction(theta, xtrain_norm(:,1:i), ctrain(:,1:i), x, post);

    Err = sigma2_y(:)+(y(:)-mu_y(:)).^2;
    score_y(i) = mean(Err);
    
    Err = var_muc(:)+(link(y(:))-mu_c(:)).^2;
    score_c(i) = mean(Err);

%     
%     score_y(i) = sqrt(mse(mu_y(:),y(:))); 
%     score_c(i) = sqrt(mse(mu_c(:),link(y(:))));
    cum_regret(i+1) = cum_regret(i)+score_c(i);
    
    
    score_c(i)= -score_c(i);
    score_y(i)= -score_y(i);
    if i > ninit
        init_guess = theta;
        theta = multistart_minConf(@(hyp)negloglike_bin(hyp, xtrain_norm(:,1:i), ctrain(:,1:i), model, 'post', post), hyp_lb, hyp_ub,10, init_guess, options_theta); 
    end
    if i> nopt              
        [new_x, new_x_norm, new_i] = acquisition_fun(x, theta,  xtrain_norm(:,1:i), ctrain(:,1:i),model, post);        
    else
        new_i = randsample(N,1);
        new_x = x(:,new_i); 
        new_x_norm = (new_x - model.lb)./(model.ub - model.lb);
    end
end
return

xp = x;
xp(2,:) = 1-xp(2,:);
figure()
subplot(1,2,1)
scatter(xp(1,:),xp(2,:), 5, normcdf(y), 'filled');
pbaspect([1,1,1])
subplot(1,2,2)
scatter(xp(1,:),xp(2,:), 5, mu_c, 'filled');
pbaspect([1,1,1])


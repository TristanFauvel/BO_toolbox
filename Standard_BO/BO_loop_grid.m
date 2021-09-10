function [x_tr, y_tr, cum_regret]= BO_loop_grid(n,maxiter, nopt, model, theta, x, y, acquisition, ninit)

idx= randsample(n,maxiter); % for random sampling


if numel(idx)==1
    i_tr = idx;
else
    i_tr= randsample(idx,1);
end
new_x = x(:,i_tr);
new_y = y(:,i_tr);

x_tr = [];
y_tr = [];

cum_regret_i =0;
cum_regret=NaN(1, maxiter+1);
cum_regret(1)=0;

if strcmp(acquisition, 'random')
    nopt= maxiter +1;
end
options_theta.method = 'lbfgs';
options_theta.verbose = 1;
ncov_hyp = numel(theta.cov);
nmean_hyp = numel(theta.mean);

theta_lb = -8*ones(ncov_hyp  + nmean_hyp ,1);
theta_lb(end) = 0;
theta_ub = 10*ones(ncov_hyp  + nmean_hyp ,1);
theta_ub(end) = 0;
regularization = 'nugget';
for i =1:maxiter
    x_tr = [x_tr, new_x];
    y_tr = [y_tr, new_y];
    cum_regret_i  =cum_regret_i + max(y)-max(y_tr);
    cum_regret(i+1) = cum_regret_i;

   
    [mu_y, sigma2_y, ~, ~, Sigma2_y]= prediction(theta, x_tr, y_tr, x, model, []);
    
    if i > ninit
        update = 'cov';       
        init_guess = [theta.cov; theta.mean];
        hyp = multistart_minConf(@(hyp)minimize_negloglike(hyp, x_tr, y_tr, kernelfun, meanfun, ncov_hyp, nmean_hyp, update), theta_lb, theta_ub,10, init_guess, options_theta); 
        theta.cov = hyp(1:ncov_hyp);
        theta.mean = hyp(ncov_hyp+1:ncov_hyp+nmean_hyp);
    end
    if i> nopt              
        sigma2_y(sigma2_y<0)=0;
        bestf = max(y_tr);
        
        if strcmp(acquisition, 'EI')
        EI  = expected_improvement(mu_y, sigma2_y,[], [], bestf, 'max');
        [max_EI, new_i] = max(EI);
        elseif strcmp(acquisition, 'TS')
            sample = mvnrnd(mu_y,Sigma2_y);
            [~, new_i] = max(sample);
        else
            error('This acquisition function is not supported')
        end
        new_x= x(:, new_i);
        new_y = y(x==new_x);
    else
        i_tr= idx(i); %random sampling
        new_x = x(:,i_tr);
        new_y = y(:,i_tr); % no noise %%%%%%%%%%%%%%%%%%%       
        
    end
end



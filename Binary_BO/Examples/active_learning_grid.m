function [x_tr, c_tr, cum_regret]= active_learning_grid(n,maxiter, nopt, kernelfun, meanfun, theta, x, y, acquisition)

idx= randsample(n,maxiter+1); % for random sampling


new_i = idx(1);
new_x = x(:,new_i);

x_tr = [];
c_tr = [];

cum_regret_i =0;
cum_regret=NaN(1, maxiter+1);
cum_regret(1)=0;

if strcmp(acquisition, 'random')
    nopt= maxiter +1;
end
modeltype  = 'exp_prop';
for i =1:maxiter
    x_tr = [x_tr, new_x];
    c_tr = [c_tr, y(new_i)>rand];
       
    [mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc]= prediction_bin(theta, x_tr, c_tr, x, kernelfun,  'modeltype', modeltype);
  
    cum_regret_i  = cum_regret_i + sqrt(mse(mu_c',y));
    cum_regret(i+1) = cum_regret_i;
    
    if i> nopt       
        if strcmp(acquisition, 'BALD')
            [~, new_i, ~] = adaptive_sampling_binary_grid(x, theta,  x_tr, c_tr, kernelfun, modeltype);
        elseif strcmp(acquisition, 'maxvar')
            [~, new_i] = max(var_muc);
        else
            error('This acquisition function is not supported')
        end
    else
        new_i= idx(i+1); %random sampling
    end
    new_x= x(:, new_i);
end
return
% figure()
% plot(mu_c); hold on;
% plot(y); hold off;
% sqrt(mean((mu_c'-y).^2))

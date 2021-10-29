function [xtrain, ytrain, cum_regret]= BO_loop_grid(n,maxiter, nopt, model, theta, x, y, acquisition, ninit)

idx= randsample(n,maxiter); % for random sampling


if numel(idx)==1
    i_tr = idx;
else
    i_tr= randsample(idx,1);
end
new_x = x(:,i_tr);
new_y = y(:,i_tr);

xtrain = [];
ytrain = [];

cum_regret_i =0;
cum_regret=NaN(1, maxiter+1);
cum_regret(1)=0;

if strcmp(acquisition, 'random')
    nopt= maxiter +1;
end

update = 'cov';

for i =1:maxiter
    xtrain = [xtrain, new_x];
    ytrain = [ytrain, new_y];
    cum_regret_i  =cum_regret_i + max(y)-max(ytrain);
    cum_regret(i+1) = cum_regret_i;


    [mu_y, sigma2_y, ~, ~, Sigma2_y]= model.prediction(theta, xtrain, ytrain, x, []);

    if i > ninit
        theta = model.model_selection(xtrain, ytrain, theta, update);
    end
    if i> nopt
        sigma2_y(sigma2_y<0)=0;
        bestf = max(ytrain);

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



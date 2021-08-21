function [xtrain, xtrain_norm, ctrain, score] = AL_preference_loop(acquisition_fun, seed, maxiter, theta, g, update_period, model, c)


xbounds = [model.lb(:),model.ub(:)];
D= size(xbounds,1);

x0 = zeros(D,1);
condition.x0 = x0;
lb = model.lb;
ub = model.ub;
base_kernelfun = model.base_kernelfun;
if c == 1
    condition.y0 = 0;
    model.kernelfun = @(theta, xi, xj, training, regularization) conditional_preference_kernelfun(theta, base_kernelfun, xi, xj, training, regularization,condition.x0);
else
    model.kernelfun = @(theta, xi, xj, training, regularization) preference_kernelfun(theta, base_kernelfun, xi, xj, training, regularization);
end

theta_init = theta;

%% Initialize the experiment
% maxiter = 200; %total number of iterations
ninit = 5; % number of time steps before starting using the acquisition function

rng(seed)
if strcmp(model.kernelname, 'Matern52') || strcmp(model.kernelname, 'Matern32') %|| strcmp(kernelname, 'ARD')
    approximation.method = 'RRGP';
else
    approximation.method = 'SSGP';
end
approximation.nfeatures = 4096;
[approximation.phi_pref, approximation.dphi_pref_dx, approximation.phi, approximation.dphi_dx]= sample_features_preference_GP(theta, D, model, approximation);

options_theta.method = 'lbfgs';
options_theta.verbose = 1;

% Warning : the seed has to be re-initialized after the random kernel
% approximation.
rng(seed)
  xduel1 =  rand_interval(lb,ub);
xduel2 =  rand_interval(lb,ub);
new_duel= [xduel1; xduel2]; %initial sample

x_best = zeros(D,maxiter);
score = zeros(1,maxiter);
model.min_x = [lb; lb];
model.max_x = [ub; ub];

xtrain = NaN(2*D, maxiter);
ctrain = NaN(1, maxiter);
 
xtest = rand_interval(lb, ub, 'nsamples', 1000);
xtest = [xtest;x0*ones(1,1000)];
xtest_norm = (xtest - [lb; lb])./([ub; ub]- [lb; lb]);

model.lb_norm = [model.lb_norm;model.lb_norm];
model.ub_norm = [model.ub_norm;model.ub_norm];

for i =1:maxiter
    disp(i)
    x_duel1 = new_duel(1:D,:);
    x_duel2 = new_duel(D+1:end,:);
    %Generate a binary sample
    c = model.link(g(x_duel1)-g(x_duel2))>rand;
    
    xtrain(:,i) = new_duel;
    ctrain(i) = c;
    
    %% Normalize data so that the bound of the search space are 0 and 1.
    xtrain_norm = (xtrain(:,1:i) - [lb; lb])./([ub; ub]- [lb; lb]);
    
    if i>ninit
        options=[];
        %Local optimization of hyperparameters
        if mod(i, update_period) ==0
            theta = theta_init(:);
            theta = minFuncBC(@(hyp)negloglike_bin(hyp, xtrain_norm(:,1:i), ctrain(1:i), model), theta, model.theta_lb, model.theta_ub, options);
        end
    end
    post =  prediction_bin(theta, xtrain_norm(:,1:i), ctrain(1:i), [], model, []);
    
    if i>ninit
        new_duel = acquisition_fun(theta, xtrain_norm(:,1:i), ctrain(1:i), model, post, approximation);
        x_duel1= new_duel(1:D);
        x_duel2 = new_duel((1+D):end);
    else %When we have not started to train the GP classification model, the acquisition is random
        [x_duel1,x_duel2]=random_acquisition_pref(theta, [], [], model, post, approximation);
    end
    new_duel = [x_duel1;x_duel2];
    
    if i == 1
        init_guess = [];
    else
        init_guess = x_best(:, end);
    end
    
    [mu_c, mu_y, sigma2_y] = prediction_bin(theta, xtrain_norm(:,1:i), ctrain(1:i), xtest_norm, model, post);
    
    gvals = g(xtest(1:model.D,:))';
    Err = sigma2_y+(gvals-mu_y).^2;
    
    score(i) = mean(Err);
end
return



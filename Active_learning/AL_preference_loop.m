function [xtrain, xtrain_norm, ctrain, score] = AL_preference_loop(acquisition_fun, seed, lb, ub, maxiter, theta, g, update_period, modeltype, theta_lb, theta_ub, kernelname, base_kernelfun, lb_norm, ub_norm, link, c)


xbounds = [lb(:),ub(:)];
D= size(xbounds,1);

x0 = zeros(D,1);
condition.x0 = x0;

if c == 1
    condition.y0 = 0;
    kernelfun = @(theta, xi, xj, training, regularization) conditional_preference_kernelfun(theta, base_kernelfun, xi, xj, training, regularization,condition.x0);
else
    kernelfun = @(theta, xi, xj, training, regularization) preference_kernelfun(theta, base_kernelfun, xi, xj, training, regularization);
end

theta_init = theta;

%% Initialize the experiment
% maxiter = 200; %total number of iterations
ninit = 5; % number of time steps before starting using the acquisition function

rng(seed)
if strcmp(kernelname, 'Matern52') || strcmp(kernelname, 'Matern32') %|| strcmp(kernelname, 'ARD')
    approximation_method = 'RRGP';
else
    approximation_method = 'SSGP';
end
nfeatures = 4096;
[kernel_approx.phi_pref, kernel_approx.dphi_pref_dx, kernel_approx.phi, kernel_approx.dphi_dx]= sample_features_preference_GP(theta, D, kernelname, approximation_method, nfeatures);

options_theta.method = 'lbfgs';
options_theta.verbose = 1;

% Warning : the seed has to be re-initialized after the random kernel
% approximation.
rng(seed)
options.method = 'lbfgs';
ncandidates= 10;
xduel1 =  rand_interval(lb,ub);
xduel2 =  rand_interval(lb,ub);
new_duel= [xduel1; xduel2]; %initial sample

x_best_norm = zeros(D,maxiter);
x_best = zeros(D,maxiter);
score = zeros(1,maxiter);
min_x = [lb; lb];
max_x = [ub; ub];

xtrain = NaN(2*D, maxiter);
ctrain = NaN(1, maxiter);
regularization = 'nugget';

xtest = rand_interval(lb, ub, 'nsamples', 1000);
xtest = [xtest;x0*ones(1,1000)];
xtest_norm = (xtest - [lb; lb])./([ub; ub]- [lb; lb]);

for i =1:maxiter
    disp(i)
    x_duel1 = new_duel(1:D,:);
    x_duel2 = new_duel(D+1:end,:);
    %Generate a binary sample
    c = link(g(x_duel1)-g(x_duel2))>rand;
    
    xtrain(:,i) = new_duel;
    ctrain(i) = c;
    
    %% Normalize data so that the bound of the search space are 0 and 1.
    xtrain_norm = (xtrain(:,1:i) - [lb; lb])./([ub; ub]- [lb; lb]);
    
    if i>ninit
        options=[];
        %Local optimization of hyperparameters
        if mod(i, update_period) ==0
            theta = theta_init(:);
            theta = minFuncBC(@(hyp)negloglike_bin(hyp, xtrain_norm(:,1:i), ctrain(1:i), kernelfun, 'modeltype', modeltype), theta, theta_lb, theta_ub, options);
        end
    end
    post =  prediction_bin(theta, xtrain_norm(:,1:i), ctrain(1:i), [], kernelfun, modeltype, [], regularization);
    
    if i>ninit
        new_duel = acquisition_fun(theta, xtrain_norm(:,1:i), ctrain(1:i), kernelfun,modeltype, max_x, min_x, [lb_norm; lb_norm], [ub_norm;ub_norm], post);
        x_duel1= new_duel(1:D);
        x_duel2 = new_duel((1+D):end);
    else %When we have not started to train the GP classification model, the acquisition is random
        [x_duel1,x_duel2]=random_acquisition_pref([],[],[],[],[],[], max_x, min_x, lb_norm, ub_norm, [], []);
    end
    new_duel = [x_duel1;x_duel2];
    
    if i == 1
        init_guess = [];
    else
        init_guess = x_best(:, end);
    end
    
    [mu_c, mu_y, sigma2_y] = prediction_bin(theta, xtrain_norm(:,1:i), ctrain(1:i), xtest_norm, kernelfun, modeltype, [], regularization);
    
    gvals = g(xtest(1:D,:))';
    Err = sigma2_y+(gvals-mu_y).^2;
    
    score(i) = mean(Err);
end
return



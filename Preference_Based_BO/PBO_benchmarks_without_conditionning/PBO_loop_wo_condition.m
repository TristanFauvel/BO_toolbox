function [xtrain, xtrain_norm, ctrain, score] = PBO_loop_wo_condition(acquisition_fun, seed, lb, ub, maxiter, theta, g, update_period, modeltype, theta_lb, theta_ub, kernelname, base_kernelfun, lb_norm, ub_norm, link)


xbounds = [lb(:),ub(:)];
D= size(xbounds,1);

x0 = zeros(D,1);
condition.x0 = x0;
% condition.y0 = 0;

kernelfun = @(theta, xi, xj, training, regularization) preference_kernelfun(theta, base_kernelfun, xi, xj, training, regularization);
theta_init = theta;

%% Initialize the experiment
% maxiter = 200; %total number of iterations
ninit = 5; % number of time steps before starting using the acquisition function

options_theta.method = 'lbfgs';
options_theta.verbose = 1;

rng(seed)
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
[kernel_approx.phi_pref, kernel_approx.dphi_pref_dx, kernel_approx.phi, kernel_approx.dphi_dx]= sample_features_preference_GP(theta, D, kernelname, approximation_method, nfeatures);


kernelfun = @(theta, xi, xj, training, regularization) conditional_preference_kernelfun(theta, base_kernelfun, xi, xj, training, regularization,condition.x0);

theta_init = theta;

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
        
        [x_duel1, x_duel2] = acquisition_fun(theta, xtrain_norm(:,1:i), ctrain(1:i), kernelfun, base_kernelfun, modeltype,max_x, min_x, lb_norm, ub_norm, condition, post, kernel_approx);
    else %When we have not started to train the GP classification model, the acquisition is random
        [x_duel1,x_duel2]=random_acquisition_pref([],[],[],[],[],[], max_x, min_x, lb_norm, ub_norm, [], []);
    end
    new_duel = [x_duel1;x_duel2];
    
    if i == 1
        init_guess = [];
    else
        init_guess = x_best(:, end);
    end
    
    x_best_norm(:,i) = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm(:,1:i), ctrain(1:i), x, kernelfun, x0,modeltype, post), lb_norm, ub_norm, ncandidates, init_guess, options);
    x_best(:,i) = x_best_norm(:,i) .*(max_x(1:D)-min_x(1:D)) + min_x(1:D);
    
    score(i) = g(x_best(:,i));
    if isnan(score(i))
        disp('bug')
    end
end
return



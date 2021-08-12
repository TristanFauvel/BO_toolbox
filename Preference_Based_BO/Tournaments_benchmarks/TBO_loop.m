function [xtrain, xtrain_norm, ctrain, score] = TBO_loop(acquisition_fun, seed, maxiter, theta, g, update_period, model, tsize,feedback)


xbounds = [lb(:),ub(:)];
D= size(xbounds,1);

x0 = zeros(D,1);
condition.x0 = x0;
condition.y0 = 0;
kernelfun = @(theta, xi, xj, training, regularization) conditional_preference_kernelfun(theta, base_kernelfun, xi, xj, training, regularization,condition.x0);

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
nfeatures = 4096;
[approximation.phi_pref, approximation.dphi_pref_dx, approximation.phi, approximation.dphi_dx]= sample_features_preference_GP(theta, D, model, approximation);
options_theta.method = 'lbfgs';
options_theta.verbose = 1;

% Warning : the seed has to be re-initialized after the random kernel
% approximation.
rng(seed)
options.method = 'lbfgs';
ncandidates= 10;

x_best_norm = zeros(D,maxiter);
x_best = zeros(D,maxiter);
score = zeros(1,maxiter);
min_x = lb;
max_x = ub;

xtrain =[];
ctrain = [];
regularization = 'nugget';

new_x =random_acquisition_tour([],[],[],[],[],[], ub, lb, lb_norm, ub_norm, [], [], [], tsize);

for i =1:maxiter
    disp(i)
    
    
    if strcmp(feedback, 'all')
        gmat = g(new_x);

        iduels = nchoosek(1:tsize,2)';
        gmat = gmat(iduels);
        c= link(gmat(1,:) - gmat(2,:)) > rand(1, size(iduels,2));
        new_xtrain = reshape(new_x(:,iduels(:)),2*D, numel(c));
    elseif strcmp(feedback, 'best')
        gmat = g(new_x);
        p = link(gmat-gmat');
        p(logical(eye(tsize)))=1;
        weights = prod(p,1);
        %normalize the weights
        weights =weights/sum(weights);        
        %Generate a binary sample
        c= mnrnd(1,weights); %sample from the corresponding categorical distribution
        
        idx = 1:tsize;
        idwin= idx(logical(c));
        idloss =idx(logical(1-c));
        
        idx = [repmat(idwin, 1, tsize-1); idloss]; % corresponding duels 
        new_xtrain = reshape(new_x(:,idx(:)),2*D, tsize-1);
        c = ones(1, tsize-1);
    end
    
    xtrain = [xtrain, new_xtrain];
    ctrain  = [ctrain, c];
    
    %% Normalize data so that the bound of the search space are 0 and 1.
    xtrain_norm = (xtrain - [lb; lb])./([ub; ub]- [lb; lb]);
    
    if i>ninit
        options=[];
        %Local optimization of hyperparameters
        if mod(i, update_period) ==0
            theta = theta_init(:);
            theta = minFuncBC(@(hyp)negloglike_bin(hyp, xtrain_norm, ctrain, model), theta, model.theta_lb, model.theta_ub, options);
        end
    end
        post =  prediction_bin(theta, xtrain_norm, ctrain, [], model, post);

    if i>ninit
        new_x = acquisition_fun(theta, xtrain_norm, ctrain, model, post, approximation, tsize);
    else %When we have not started to train the GP classification model, the acquisition is random
        new_x =random_acquisition_tour([],[],[],[],[],[], max_x, min_x, lb_norm, ub_norm, [], [],[], tsize);
    end
    
    
    if i == 1
        init_guess = [];
    else
        init_guess = x_best(:, end);
    end
    
    x_best_norm(:,i) = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);
    x_best(:,i) = x_best_norm(:,i) .*(model.ub(1:D)-model.lb(1:D)) + model.lb(1:D);
    
    score(i) = g(x_best(:,i));
    if isnan(score(i))
        disp('bug')
    end
end
return


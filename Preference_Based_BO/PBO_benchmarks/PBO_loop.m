function [xtrain, xtrain_norm, ctrain, score] = PBO_loop(acquisition_fun, seed, lb, ub, maxiter, theta, g, update_period, modeltype, theta_lb, theta_ub, kernelname, base_kernelfun, lb_norm, ub_norm, link)


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
if any(strcmp(func2str(acquisition_fun), {'DTS', 'kernelselfsparring', 'Thompson_challenge'}))
    if strcmp(kernelname, 'Matern52') || strcmp(kernelname, 'Matern32') %|| strcmp(kernelname, 'ARD')
        approximation.method = 'RRGP';
    else
        approximation.method = 'SSGP';
    end
    nfeatures = 4096;
    %approximation.phi : ntest x nfeatures
    %approximation.phi_pref : ntest x nfeatures
    % approximation.dphi_dx : nfeatures x D
    % approximation.dphi_dx : nfeatures x 2D
    [approximation.phi_pref, approximation.dphi_pref_dx, approximation.phi, approximation.dphi_dx]= sample_features_preference_GP(theta, D, model, approximation);
else
    approximation = [];
end
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
            theta = minFuncBC(@(hyp)negloglike_bin(hyp, xtrain_norm(:,1:i), ctrain(1:i), model), theta, theta_lb, theta_ub, options);
        end
    end
        post =  prediction_bin(theta, xtrain_norm(:,1:i), ctrain(1:i), [], model, post);

    if i>ninit
        [x_duel1, x_duel2] = acquisition_fun(theta, xtrain_norm(:,1:i), ctrain(1:i), model modeltype,max_x, min_x, lb_norm, ub_norm, condition, post, approximation);
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

figure()
plot(score)


if d == 1
    N = 100;
    
    x = linspace(0,1,N);
    y = g(x)';
    
    figure();
    plot(x, y)
    [d,n]= size(x);
    kernelname = kernelname;
    
    [~,  g_mu_y, g_sigma2_y, g_Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx] = prediction_bin(theta, xtrain_norm, ctrain, [x;x0*ones(1,N)], kernelfun,kernelname, modeltype, post, regularization);
    figure();
    errorshaded(x, g_mu_y, sqrt(g_sigma2_y));
    
    figure()
    imagesc(x,x,g_mu_y-g_mu_y'); hold on;
    scatter(xtrain(1,:), xtrain(2,:), 25, 'k', 'filled');hold off
    pbaspect([1 1 1])
    set(gca,'YDir','normal')
    
    figure()
    imagesc(x,x,y-y');
    pbaspect([1 1 1])
    set(gca,'YDir','normal')
    
    
    result = NaN(2*d,N^d);
    num_result = NaN(d,N^d);
    k=@(x) to_test_prediction_bin(theta, xtrain_norm, ctrain, x,x0, kernelfun,kernelname, modeltype)
    
    for i = 1:N^d
        num_result(:,i) = test_deriv(k, x(:,i), 1e-12);
        [~, ~,~,~,~,~, result(:,i)] = prediction_bin(theta, xtrain_norm, ctrain, [x(:,i);x0.*ones(d,1)], kernelfun,kernelname, modeltype, post, regularization);
    end
    result = result(1:d,:);
    figure()
    plot(result(:)); hold on;
    plot(real(num_result(:))); hold off;
    
    
    
    [EI, dEI_dx] = expected_improvement_for_classification(theta, xtrain_norm, x, ctrain, lb_norm, ub_norm, kernelfun,kernelname, x0, modeltype)
    
    figure()
    plot(x,-EI)
    
elseif D==2
    N = 22;
    xrange = linspace(0,1,N);
    [p,q] = ndgrid(xrange, xrange);
    x2d= [p(:),q(:)]';
    [~,  g_mu_y, g_sigma2_y, g_Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx] = prediction_bin(theta, xtrain_norm, ctrain, [x2d;x0.*ones(d,N^d)], kernelfun,kernelname, modeltype, post, regularization);
    fig=figure()
    fig.Color =  [1 1 1];
    imagesc(xrange,xrange,reshape(g_mu_y, N,N)); hold on;
    set(gca,'YDir','normal')
    h=colorbar()
    xlabel('x2')
    ylabel('x1')
    col = linspace(h.Limits(1),h.Limits(2),i)
    scatter(x_best(2,1:i),x_best(1,1:i), 25, col,'filled'); hold off;
    pbaspect([1 1 1])
    title('Inferred value function')
    
    true_values = g(x2d);
    fig=figure()
    fig.Color =  [1 1 1];
    imagesc(xrange, xrange, reshape(true_values, N,N))
    pbaspect([1 1 1])
    colorbar()
    set(gca,'YDir','normal')
    title('True value function')
    xlabel('x2')
    ylabel('x1')
    
    fig=figure()
    fig.Color =  [1 1 1];
    imagesc(xrange,xrange,reshape(g_sigma2_y, N,N))
    pbaspect([1 1 1])
    colorbar()
    set(gca,'YDir','normal')
    xlabel('x2')
    ylabel('x1')
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    n_candidates = 250
    l= zeros(d,n_candidates);
    for i = 1:n_candidates
        l(:,i) = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, kernelfun, x0,modeltype), lb_norm, ub_norm, 1, options);
    end
    fig=figure()
    fig.Color =  [1 1 1];
    imagesc(xrange,xrange,reshape(g_mu_y, N,N)); hold on;
    set(gca,'YDir','normal')
    h=colorbar()
    xlabel('x2')
    ylabel('x1')
    scatter(l(2,1:i),l(1,1:i), 25, 'k','filled'); hold off;
    pbaspect([1 1 1])
    title('Inferred value function')
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    k=@(x) to_test_prediction_bin(theta, xtrain_norm, ctrain, x,x0, kernelfun,kernelname, modeltype)
    
    result = NaN(2*d,N^d);
    num_result = NaN(d,N^d);
    
    for i = 1:N^d
        num_result(:,i) = test_deriv(k, x2d(:,i), 1e-12);
        [~, ~,~,~,~,~, result(:,i)] = prediction_bin(theta, xtrain_norm, ctrain, [x2d(:,i);x0.*ones(d,1)], kernelfun,kernelname, modeltype, post, regularization);
    end
    result = result(1:d,:);
    figure()
    plot(result(:)); hold on;
    plot(real(num_result(:))); hold off;
    
    
    figure(); plot(sqrt(g_sigma2_y))
    
    %%
    
    
    
    
    figure(); plot(sqrt(g_sigma2_y))
    
end
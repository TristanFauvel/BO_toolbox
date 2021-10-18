function [xtrain, xtrain_norm, ctrain, score_c, score_g,x_best_c, x_best_g]= BBO_loop(acquisition_fun, nopt, seed, maxiter, theta, g, update_period, model)

% g : objective function
% maxiter : number of iterations
% nopt : number of time steps before starting using the acquisition
% ninit : number of time steps before starting updating the hyperparameters
% function

ub = model.ub;
lb= model.lb;
lb_norm = model.lb_norm;
ub_norm = model.ub_norm;

D = numel(ub);

xtrain = [];
xtrain_norm = [];
ctrain = [];


options_theta.method = 'lbfgs';
options_theta.verbose = 1;

rng(seed)
new_x_norm = rand_interval(lb_norm,ub_norm);
new_x = new_x_norm.*(ub - lb)+lb;
ninit= maxiter + 2;

options.method = 'lbfgs';
ncandidates= 10;
%% Compute the kernel approximation if needed
if strcmp(model.kernelname, 'Matern52') || strcmp(model.kernelname, 'Matern32') || strcmp(model.kernelname, 'ARD')
    approximation.method = 'RRGP';
else
    approximation.method = 'SSGP';
end
approximation.decoupled_bases = 1;
approximation.nfeatures = 256;
[approximation.phi, approximation.dphi_dx] = sample_features_GP(theta(:), model, approximation);
x_best_norm_c = zeros(D, maxiter);
x_best_c = zeros(D, maxiter);
x_best_norm_g = zeros(D, maxiter);
x_best_g = zeros(D, maxiter);

score = zeros(1,maxiter);


identification = 'mu_c';
for i =1:maxiter
    disp(i)
    new_c = model.link(g(new_x))>rand;
    xtrain = [xtrain, new_x];
    xtrain_norm = [xtrain_norm, new_x_norm];
    ctrain = [ctrain, new_c];
    
    
    if i > ninit
        %Local optimization of hyperparameters
        if mod(i, update_period) ==0
            init_guess = theta;
            theta = multistart_minConf(@(hyp)minimize_negloglike_bin(hyp, xtrain_norm, ctrain, kernelfun, meanfun, update, post), hyp_lb, hyp_ub,10, init_guess, options_theta);
            [approximation.phi, approximation.dphi_dx] = sample_features_GP(theta(:), model, approximation);
            
        end
    end
    post =  model.prediction(theta, xtrain_norm, ctrain, [], model, []);
    
    if i> nopt
        [new_x, new_x_norm] = acquisition_fun(theta, xtrain_norm, ctrain,model, post, approximation);
    else
        new_x_norm = rand_interval(lb_norm,ub_norm);
        new_x = new_x_norm.*(ub - lb)+lb;
    end
    init_guess = [];
    
    %     if strcmp(identification, 'mu_c')
    x_best_norm_c(:,i) = multistart_minConf(@(x)to_maximize_mu_c_GP(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);
    
    %     elseif strcmp(identification, 'mu_g')
    x_best_norm_g(:,i) = multistart_minConf(@(x)to_maximize_mean_bin_GP(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);
    %     end
    
    x_best_c(:,i) = x_best_norm_c(:,i) .*(ub-lb) + lb;
    x_best_g(:,i) = x_best_norm_g(:,i) .*(ub-lb) + lb;
    
    score_c(i) = normcdf(g(x_best_c(:,i)));
    score_g(i) = g(x_best_g(:,i));
    
%     if i == 200
%         disp('stop')
%     end
    
    
end
return

%%
graphics_style_paper
xx = linspace(ub, lb, 100);
xx_norm = linspace(ub_norm, lb_norm, 100);
[mu_c,  mu_y, sigma2_y, Sigma2_y] =  model.prediction(theta, xtrain_norm, ctrain, xx_norm, post);
figure()
plot_gp(xx, mu_y, sigma2_y, C(1,:), 2); hold on;
plot(xx, g(xx));
scatter(xtrain, ctrain, markersize, 'k', 'filled');

Y = normcdf(mvnrnd(mu_y, Sigma2_y,10000));
figure()
[p1,p2, h] = plot_distro(xx, mu_c, Y, C(1,:), C(2,:),linewidth); hold on
scatter(xtrain, ctrain, markersize, 'k', 'filled');
plot(xx, normcdf(g(xx)), 'color', 'k')


samples_prior = mvnrnd(zeros(1,100), model.kernelfun(theta, xx,xx,[], 'none'), 10);
figure()
plot(samples_prior')
%%
init_guess = theta;
theta = multistart_minConf(@(hyp)negloglike_bin(hyp, xtrain_norm, ctrain, model), model.hyp_lb, model.hyp_ub,10, init_guess, options_theta);
[mu_c,  mu_y, sigma2_y, Sigma2_y] =  model.prediction(theta, xtrain_norm, ctrain, xx_norm, post);

figure()
plot_gp(xx, mu_y, sigma2_y, C(1,:), 2); hold on;
plot(xx, g(xx));
scatter(xtrain, ctrain, markersize, 'k', 'filled');

Y = normcdf(mvnrnd(mu_y, Sigma2_y,10000));
figure()
[p1,p2, h] = plot_distro(xx, mu_c, Y, C(1,:), C(2,:),linewidth); hold on
scatter(xtrain, ctrain, markersize, 'k', 'filled');
plot(xx, normcdf(g(xx)), 'color', 'k')


ytrain = g(xtrain);
hyp.cov = theta;
hyp.mean = 0;
[mu_y, sigma2_y] = prediction(hyp, xtrain_norm, ytrain, xx_norm, model, []);
figure()
plot_gp(xx, mu_y, sigma2_y, C(1,:), 2); hold on;
plot(xx, g(xx));
scatter(xtrain_norm, ytrain, markersize, 'k', 'filled');



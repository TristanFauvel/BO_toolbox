function [x_duel1, x_duel2, new_duel] = bivariate_EI(theta, xtrain_norm, ctrain, model, post, approximation)
%'Bivariate EI only possible with duels, not tournaments'
% Bivariate Expected Improvement, as proposed by Nielsen (2015)
%% Find the maximum of the value function
options.method = 'lbfgs';
options.verbose = 1;

D = size(xtrain_norm,1)/2;
n = size(xtrain_norm,2);
ncandidates =5;
init_guess = [];

% x_duel1 = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm, model.ub_norm, ncandidates,init_guess, options);
x = [xtrain_norm(1:D,:), xtrain_norm((D+1):end,:)];

[g_mu_c,  g_mu_y] = prediction_bin(theta, xtrain_norm, ctrain, [x;model.condition.x0*ones(1,2*n)], model, post);
[a,b]= max(g_mu_y);
x_duel1 = x(:,b);

x_duel2 = multistart_minConf(@(x)compute_bivariate_expected_improvement(theta, xtrain_norm, x, ctrain, model, x_duel1, post), model.lb_norm, model.ub_norm, ncandidates,init_guess, options);
x_duel1 = x_duel1.*(model.max_x(1:D)-model.max_x(1:D)) + model.max_x(1:D);
x_duel2 = x_duel2.*(model.max_x(D+1:2*D)-model.min_x(D+1:2*D)) + model.min_x(D+1:2*D);

new_duel = [x_duel1;x_duel2];

end

function [BEI, dBEI_dx] = compute_bivariate_expected_improvement(theta, xtrain_norm, x, ctrain, model, x_duel1, post)

[D,n]= size(x);
x0 = model.condition.x0;
[~, ~, g_sigma2_y,  ~, ~, dmuy_dx] = prediction_bin(theta, xtrain_norm, ctrain, [x;x0*ones(1,n)], model, post);

dmuy_dx = dmuy_dx(1:D,:);

g_sigma_y = sqrt(g_sigma2_y);


% Compute the joint predictive distribution of the best point (first duel member) and the candidate
[~, g_mu_y, g_sigma2_y, g_Sigma2_y, ~, ~, ~, dSigma2_y_dx] = prediction_bin(theta, xtrain_norm, ctrain, [x_duel1, x;x0*ones(1,n),x0*ones(1,n)], model, post);

g_sigma2_y = g_sigma2_y(1);
max_mu_y = g_mu_y(1);
g_mu_y = g_mu_y(2);

sigma_y = sqrt(g_sigma2_y);

sigma_2_I = g_Sigma2_y(1,1) + g_Sigma2_y(2,2) - 2*g_Sigma2_y(1,2);
sigma_I=sqrt(sigma_2_I);

d = (g_mu_y - max_mu_y)./sigma_I;
d(sigma_y==0)=0;

normpdf_d =  normpdf(d);
normcdf_d = normcdf(d);

BEI = (g_mu_y - max_mu_y).*normcdf_d+ sigma_I.*normpdf_d;
BEI(sigma_y==0) = 0;

if nargout>1   

    gaussder_d = -d.*normpdf_d; %derivative of the gaussian
    
    dsigma2_I_dx = squeeze(dSigma2_y_dx(2,2,2,1:D) - 2*dSigma2_y_dx(1,2,2,1:D));
    dsigma_I_dx= dsigma2_I_dx./(2*sigma_I);
    
    dd_dx = (-dmuy_dx.*sigma_I - (max_mu_y - g_mu_y).*dsigma_I_dx)./sigma_2_I;
    dd_dx(g_sigma2_y==0,:) = 0;
    dBEI_dx = dmuy_dx.*normcdf_d - (max_mu_y - g_mu_y).*normpdf_d.*dd_dx + dsigma_I_dx.*normpdf_d +g_sigma_y.*gaussder_d.*dd_dx;
    dBEI_dx = -squeeze(dBEI_dx);%This is because the goal is to maximize EI, and I use a minimization algorithm
end

BEI = -BEI; %This is because the goal is to maximize EI, and I use a minimization algorithm

end

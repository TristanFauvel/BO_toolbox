function [new_x, new_x_norm, L] = bivariate_EI_binary(theta, xtrain_norm, ctrain,model, post, ~)
% Inspired by Bivariate Expected Improvement, as proposed by Nielsen (2015)

% note that this function works in the latent space
options.method = 'lbfgs';
options.verbose = 1;

ncandidates= 5;
init_guess = [];

init_guess = [];
options.method = 'lbfgs';
options.verbose = 1;
ncandidates = 5;
%% Find the maximum of the latent function
[xbest, ybest] = multistart_minConf(@(x)to_maximize_mean_bin_GP(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm,  model.ub_norm, ncandidates, init_guess, options);

[new_x_norm,L] = multistart_minConf(@(x)compute_bivariate_expected_improvement(theta, xtrain_norm, x, ctrain, model, xbest, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);
 new_x = new_x_norm.*(model.ub-model.lb) + model.lb;
 L=-L;
end


function [BEI, dBEI_dx] = compute_bivariate_expected_improvement(theta, xtrain_norm, x, ctrain, model, xbest, post)

[D,n]= size(x);
[~, ~, g_sigma2_y,  ~, ~, dmuy_dx] = prediction_bin(theta, xtrain_norm, ctrain, x, model, post);

g_sigma_y = sqrt(g_sigma2_y);
 
[~, g_mu_y, g_sigma2_y, g_Sigma2_y] = prediction_bin(theta, xtrain_norm, ctrain, [xbest, x], model, post);

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

BEI = (g_mu_y - max_mu_y).*normcdf_d+ sigma_I.*normpdf_d;%Brochu
BEI(sigma_y==0) = 0;

if nargout>1   
    [~, ~, ~, g_Sigma2_y, ~, ~, ~, dSigma2_y_dx] = prediction_bin(theta, xtrain_norm, ctrain, [xbest, x], model, post);

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

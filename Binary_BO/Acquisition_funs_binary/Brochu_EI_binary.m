function new_x = Brochu_EI(theta, xtrain_norm, ctrain, kernelfun, modeltype, max_x, min_x, lb_norm, ub_norms, post, ~)
% Binary Expected Improvement, as proposed by Brochu (2010)
%% Find the maximum of the value function
options.method = 'sd';

ncandidates= 5;
x_init = [];
x_best = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, kernelfun, x0,modeltype, post), lb_norm, ub_norm, ncandidates, x_init, options);

x_init = [];
new_x = multistart_minConf(@(x)expected_improvement_preference(theta, xtrain_norm, x, ctrain, lb_norm, ub_norm, kernelfun,x0, x_best, modeltype), lb_norm, ub_norm, ncandidates, x_init, options, post);

d= size(new_x,1);
new_x = new_x.*(max_x(1:d)-min_x(1:d)) + min_x(1:d);

end

function [EI, dEI_dx] = expected_improvement_preference(theta, xtrain_norm, x, ctrain, lb_norm, ub_norm, kernelfun, x0, x_best, modeltype, post)

[nd,n]= size(x);
[g_mu_c,  g_mu_y, g_sigma2_y, g_Sigma2_y, dmuc_dx, dmuy_dx, dsigma2_y_dx] = prediction_bin(theta, xtrain_norm, ctrain, [x;x0*ones(1,n)], kernelfun,kernelname, 'modeltype', modeltype, 'post', post);

dmuc_dx = dmuc_dx(1:nd,:);
dmuy_dx = dmuy_dx(1:nd,:);
dsigma2_y_dx = dsigma2_y_dx(1:nd,:);

g_sigma_y = sqrt(g_sigma2_y);
%% Find the maximum of the value function
[~,  max_mu_y] = prediction_bin(theta, xtrain_norm, ctrain, [x_best;x0], kernelfun, 'modeltype', modeltype, 'post', post);


sigma_y = sqrt(g_sigma2_y);
d = (g_mu_y - max_mu_y)./sigma_y;
d(sigma_y==0)=0;

normpdf_d =  normpdf(d);
normcdf_d= normcdf(d);

EI = (g_mu_y - max_mu_y).*normcdf_d+ sigma_y.*normpdf_d;%Brochu

EI(sigma_y==0)= 0;
if nargout>1
    gaussder_d = -d.*normpdf_d; %derivative of the gaussian
    dsigma_y_dx = dsigma2_y_dx./(2*g_sigma_y);
    dsigma_y_dx(g_sigma2_y==0,:) = 0;
    dd_dx = (-dmuy_dx.*g_sigma_y - (max_mu_y - g_mu_y).*dsigma_y_dx)./g_sigma2_y;
    dd_dx(g_sigma2_y==0,:) = 0;
    dEI_dx = dmuy_dx.*normcdf_d - (max_mu_y - g_mu_y).*normpdf_d.*dd_dx + dsigma_y_dx.*normpdf_d +g_sigma_y.*gaussder_d.*dd_dx;
    dEI_dx = -squeeze(dEI_dx);%This is because the goal is to maximize EI, and I use a minimization algorithm
end
EI = -EI; %This is because the goal is to maximize EI, and I use a minimization algorithm
end

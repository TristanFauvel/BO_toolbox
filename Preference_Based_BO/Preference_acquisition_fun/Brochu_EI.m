function [x_duel1, x_duel2, new_duel] = Brochu_EI(theta, xtrain_norm, ctrain, model, post, approximation)
% Expected Improvement, as proposed by Brochu (2010)

D = size(xtrain_norm,1)/2;
n = size(xtrain_norm,2);
%% Find the maximum of the value function
options.method = 'lbfgs';

ncandidates= 5;

condition = model.condition;

x = [xtrain_norm(1:D,:), xtrain_norm((D+1):end,:)];
[g_mu_c,  g_mu_y] = prediction_bin(theta, xtrain_norm, ctrain, [x;condition.x0*ones(1,2*n)], model, post);
[max_mu_y,b]= max(g_mu_y);
x_duel1 = x(:,b);


% x_duel2 = minFuncBC(@(x)expected_improvement_for_classification(theta, xtrain_norm, x, ctrain, lb_norm, ub_norm, kernelfun,kernelname, x0, modeltype), x_init, lb_norm, ub_norm, options);
% x_duel2 = multistart_minfuncBC(@(x)expected_improvement_for_classification(theta, xtrain_norm, x, ctrain, lb_norm, ub_norm, kernelfun,kernelname, x0, x_best, modeltype), lb_norm, ub_norm, ncandidates, options);

init_guess = x_duel1;
x_duel2 = multistart_minConf(@(x)expected_improvement_preference(theta, xtrain_norm, x, ctrain, max_mu_y, model, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);

x_duel1 = x_duel1.*(model.max_x(1:D)-model.min_x(1:D)) + model.min_x(1:D);
x_duel2 = x_duel2.*(model.max_x(D+1:2*D)-model.min_x(D+1:2*D)) + model.min_x(D+1:2*D);

new_duel = [x_duel1;x_duel2];

end

function [EI, dEI_dx] = expected_improvement_preference(theta, xtrain_norm, x, ctrain, max_mu_y, model, post)

[D,n]= size(x);
[g_mu_c,  g_mu_y, g_sigma2_y, g_Sigma2_y, dmuc_dx, dmuy_dx, dsigma2_y_dx] = prediction_bin(theta, xtrain_norm, ctrain, [x;model.condition.x0*ones(1,n)], model, post);

dmuc_dx = dmuc_dx(1:D,:);
dmuy_dx = dmuy_dx(1:D,:);
dsigma2_y_dx = dsigma2_y_dx(1:D,:);

g_sigma_y = sqrt(g_sigma2_y);
%% Find the maximum of the value function

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

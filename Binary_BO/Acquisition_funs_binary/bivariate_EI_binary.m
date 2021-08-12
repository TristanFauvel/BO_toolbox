function new_x = bivariate_EI_binary(theta, xtrain_norm, ctrain,model, post, ~)

%'Bivariate EI only possible with duels, not tournaments'
% Bivariate Expected Improvement, as proposed by Nielsen (2015)
%% Find the maximum of the value function
options.method = 'sd';
options.verbose = 1;

ncandidates= 5;
x_init = [];
% x_best = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, kernelfun,kernelname, x0,modeltype), lb_norm, ub_norm, ncandidates, x_init, options);


x_best = multistart_minConf(@(x)to_maximize_activation(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm, model.ub_norm, ncandidates, x_init, options);


new_x = multistart_minConf(@(x)compute_bivariate_expected_improvement(theta, xtrain_norm, x, ctrain, model, x_best, post), model.lb_norm, model.ub_norm, ncandidates, x_init, options);
d= size(new_x,1);
new_x = new_x.*(model.ub(1:D)-model.lb(1:D)) + model.lb(1:D);
end

function  [g_mu_y,  dmuy_dx] =  to_maximize_activation(theta, xtrain_norm, ctrain, x, model, post)
regularization = 'nugget';
    [~,  g_mu_y, ~, ~, ~, dmuy_dx] = prediction_bin(theta, xtrain_norm, ctrain, x, model, post);
    g_mu_y = -g_mu_y;
    % dmuy_dx= -squeeze(dmuy_dx(:,:,1:d));
    dmuy_dx= -squeeze(dmuy_dx);

end


function [BEI, dBEI_dx] = compute_bivariate_expected_improvement(theta, xtrain_norm, x, ctrain, model, x_best, post)

[nd,n]= size(x);

%% Find the maximum of the value function

[~, g_mu_y, g_sigma2_y, g_Sigma2_y] = prediction_bin(theta, xtrain_norm, ctrain, [x_norm; model.condition.x0], model, post);

g_sigma2_y = g_sigma2_y(1);
g_sigma_y = sqrt(g_sigma2_y);

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
    [~, ~, ~, g_Sigma2_y, ~, ~, dmuy_dx, dSigma2_y_dx] = prediction_bin(theta, xtrain_norm, ctrain, x_best, model, post);
    gaussder_d = -d.*normpdf_d; %derivative of the gaussian
    
    dsigma2_I_dx = squeeze(dSigma2_y_dx(2,2,2,1:nd) - 2*dSigma2_y_dx(1,2,2,1:nd));
    dsigma_I_dx= dsigma2_I_dx./(2*sigma_I);
    
    dd_dx = (-dmuy_dx.*sigma_I - (max_mu_y - g_mu_y).*dsigma_I_dx)./sigma_2_I;
    dd_dx(g_sigma2_y==0,:) = 0;
    dBEI_dx = dmuy_dx.*normcdf_d - (max_mu_y - g_mu_y).*normpdf_d.*dd_dx + dsigma_I_dx.*normpdf_d +g_sigma_y.*gaussder_d.*dd_dx;
    dBEI_dx = -squeeze(dBEI_dx);%This is because the goal is to maximize EI, and I use a minimization algorithm
end

BEI = -BEI; %This is because the goal is to maximize EI, and I use a minimization algorithm

end

% 
% if d==1
%     N=100;
%     input = linspace(0,1,N);
%     ei = zeros(1,N);
%     dei = zeros(d,N);
% 
%     for i = 1:N
%         [EI, dEI_dx] = compute_bivariate_expected_improvement(theta, xtrain_norm, input(i), ctrain, lb_norm, ub_norm, kernelfun,kernelname, x0, x_best,modeltype);
%         ei(i) = EI;
%         dei(i) = dEI_dx;
%     end
%     figure()
%     plot(input,-ei)
%     
%      figure()
%     plot(input,-dei)
%     
%     [~,  g_mu_y, g_sigma2_y, g_Sigma2_y] = prediction_bin(theta, xtrain_norm, ctrain, [input;x0*ones(1,N)], kernelfun,kernelname, modeltype, post, regularization);
%     
%     figure()
%     errorshaded(input, g_mu_y, sqrt(g_sigma2_y))
%     
%     [EI, dEI_dx] = compute_bivariate_expected_improvement(theta, xtrain_norm, linspace(0,1,100), ctrain, lb_norm, ub_norm, kernelfun,kernelname, x0, x_best,modeltype)
% else
%     N=100;
% 
%     input =x_best*ones(1,N);    
% %     input =rand(d,N);
% 
%     di=1;
%     input(di,:) = linspace(0,1,N);
% 
%     ei = zeros(1,N);
%     dei = zeros(d,N);
%     for i = 1:N
%         [EI, dEI_dx] = compute_bivariate_expected_improvement(theta, xtrain_norm, input(:,i), ctrain, lb_norm, ub_norm, kernelfun,kernelname, x0, x_best,modeltype);
%         ei(i) = EI;
%         dei(:,i) = dEI_dx;
%     end
%     figure()
%     plot(input(di,:), -ei)
%     [~,  g_mu_y, g_sigma2_y, g_Sigma2_y] = prediction_bin(theta, xtrain_norm, ctrain, [input;x0*ones(1,N)], kernelfun,kernelname, modeltype, post, regularization);
% %         [~,  g_mu_y, g_sigma2_y, g_Sigma2_y] = prediction_bin(theta, xtrain_norm, ctrain, [input(:,i);x0], kernelfun,kernelname, modeltype, post, regularization);
% 
%     figure()
%     errorshaded(input(di,:), g_mu_y, sqrt(g_sigma2_y))
%     
%     figure()
%     plot(input(di,:), g_mu_y)
% 
%     
%     figure()
%     plot(sqrt(g_sigma2_y))
%     
%     for j = 1:d
%         figure()
%         plot(input(j,:), -dei(j,:))
%     end
%     
%     k=@(x) compute_bivariate_expected_improvement(theta, xtrain_norm, x, ctrain, lb_norm, ub_norm, kernelfun,kernelname, x0, x_best,modeltype);
%     result = NaN(d,N);
%     for i = 1:N
%         result(:,i) = test_matrix_deriv(k, input(:,i), 1e-9);
%     end
%     
%     for j = 1:d
%         figure()
%         plot(input(j,:), result(j,:))
%     end
%     
%     [EI, dEI_dx] = compute_bivariate_expected_improvement(theta, xtrain_norm, input, ctrain, lb_norm, ub_norm, kernelfun,kernelname, x0, x_best,modeltype)
% 
% 
% end

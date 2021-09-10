function [BEI, dBEI_dx] = compute_bivariate_expected_improvement(theta, xtrain_norm, x, ctrain, model, x_duel1, post)

[D,n]= size(x);
x0 = model.condition.x0;
[~, ~, g_sigma2_y,  ~, ~, dmuy_dx] = prediction_bin(theta, xtrain_norm, ctrain, [x;x0*ones(1,n)], model, post);

dmuy_dx = dmuy_dx(1:D,:);

g_sigma_y = sqrt(g_sigma2_y);


% Compute the joint predictive distribution of the best point (first duel member) and the candidate
BEI = NaN(1,n);
for i = 1:n
    [~, g_mu_y, g_sigma2_y, g_Sigma2_y, ~, ~, ~, dSigma2_y_dx] = ...
        prediction_bin(theta, xtrain_norm, ctrain, [x_duel1, x(:,i);x0,x0], model, post);
    
    g_sigma2_y = g_sigma2_y(1);
    max_mu_y = g_mu_y(1);
    g_mu_y = g_mu_y(2);
    
    %     sigma_y = sqrt(g_sigma2_y);
    
    sigma_2_I = g_Sigma2_y(1,1) + g_Sigma2_y(2,2) - 2*g_Sigma2_y(1,2);
    sigma_I=sqrt(sigma_2_I);
    
    d = (g_mu_y - max_mu_y)./sigma_I;
    d(sigma_I==0)=0;
    
    normpdf_d =  normpdf(d);
    normcdf_d = normcdf(d);
    %     if sigma_y==0
    %         BEI(i) = 0;
    %     else
    BEI(i) = (g_mu_y - max_mu_y).*normcdf_d+ sigma_I.*normpdf_d;
    %         end
end
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


function [new_x, new_x_norm] = EI(theta, xtrain_norm, ytrain, meanfun, kernelfun, kernelname, max_x, min_x, lb_norm, ub_norm, kernel_approx)

options.method = 'lbfgs';
mu_y =  prediction(theta, xtrain_norm, ytrain, xtrain_norm, kernelfun, meanfun);
y_best = max(mu_y);

x_init = [];
ncandidates = 10;
new_x_norm = multistart_minConf(@(x)expected_improvement(theta, xtrain_norm, x, ytrain, kernelfun,meanfun, y_best), lb_norm, ub_norm, ncandidates, x_init, options);
new_x = new_x_norm.*(max_x-min_x) + min_x;

end

function [EI, dEI_dx] = expected_improvement(theta, xtrain_norm, x, ytrain, kernelfun, meanfun, y_best)

[mu_y, sigma2_y,dmu_dx, dsigma2_dx] =  prediction(theta, xtrain_norm, ytrain, x, kernelfun, meanfun);

sigma_y = sqrt(sigma2_y);
d = (mu_y - y_best)./sigma_y;
d(sigma_y==0)=0;

normpdf_d =  normpdf(d);
normcdf_d= normcdf(d);

EI = (mu_y - y_best).*normcdf_d+ sigma_y.*normpdf_d;%Brochu

EI(sigma_y==0)= 0;
if nargout>1
    gaussder_d = -d.*normpdf_d; %derivative of the gaussian
    dsigma_y_dx = dsigma2_dx./(2*sigma_y);
    dsigma_y_dx(sigma2_y==0,:) = 0;
    dd_dx = (-dmu_dx.*sigma_y - (y_best - mu_y).*dsigma_y_dx)./sigma2_y;
    dd_dx(sigma2_y==0,:) = 0;
    dEI_dx = dmu_dx.*normcdf_d - (y_best - mu_y).*normpdf_d.*dd_dx + dsigma_y_dx.*normpdf_d +sigma_y.*gaussder_d.*dd_dx;
    dEI_dx = -squeeze(dEI_dx);%This is because the goal is to maximize EI, and I use a minimization algorithm
end
    EI = -EI; %This is because the goal is to maximize EI, and I use a minimization algorithm
end
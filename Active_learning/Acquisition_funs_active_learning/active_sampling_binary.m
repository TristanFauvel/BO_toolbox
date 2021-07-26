function [new_x, new_x_norm] = active_sampling_binary(theta, xtrain_norm, ctrain, kernelfun,modeltype, max_x, min_x, lb_norm, ub_norm, post)
ncandidates = 10;
init_guess = [];
options.method = 'lbfgs';
options.verbose = 1;
new_x_norm = multistart_minConf(@(x)adaptive_sampling_binary(x, theta, xtrain_norm, ctrain, kernelfun, modeltype, post), lb_norm, ub_norm, ncandidates,init_guess, options);
new_x = new_x_norm.*(max_x-min_x) + min_x;
end

function [I, dIdx]= adaptive_sampling_binary(x, theta, xtrain, ctrain, kernelfun, modeltype, post)
regularization = 'nugget';
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx] =  prediction_bin(theta, xtrain, ctrain, x, kernelfun, modeltype, post, regularization);

h = @(p) -p.*log(p+eps) - (1-p).*log(1-p+eps);


if strcmp(modeltype, 'exp_prop')
    % for a gaussian cdf link function:
    C = sqrt(pi*log(2)/2);
    
    I1 = h(mu_c);
    I2 =  log(2)*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
   
elseif strcmp(modeltype, 'laplace')
    %for a sigmoid link
    C = sqrt(2*log(2));
    I1 = h(mu_c) ;
    I2 = 2*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
    
end
    I = I1 - I2;

dhdp = @(p) -log(p)+log(1-p);
% arg = mu_y./sqrt(sigma2_y+C^2);
% dI1dx = (((dmuy_dx).*sqrt(sigma2_y+C^2)-mu_y.*dsigma2y_dx./(2*sqrt(sigma2_y+C^2)))./(sigma2_y+1)).*normpdf(arg).*dhdp(normcdf(arg));
dI1dx = dhdp(mu_c)*dmuc_dx;

dI2dx =I2.*(0.5*mu_y.^2.*dsigma2y_dx-mu_y.*(sigma2_y+C^2).*dmuy_dx)./((sigma2_y+C^2).^2)-I2./(2*(sigma2_y+C^2)).*dsigma2y_dx;

dIdx = dI1dx - dI2dx;


I = -I;
dIdx = -dIdx;
end

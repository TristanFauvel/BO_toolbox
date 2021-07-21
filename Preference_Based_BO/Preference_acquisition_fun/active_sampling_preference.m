function new_x = active_sampling_preference(theta, xtrain_norm, ctrain, kernelfun,modeltype, max_x, min_x, lb_norm, ub_norm, post)
ncandidates = 10;
init_guess = [];
options.method = 'lbfgs';
options.verbose = 1;
new_x = multistart_minConf(@(x)adaptive_sampling_preference(x, theta, xtrain_norm, ctrain, kernelfun, modeltype, post), lb_norm, ub_norm, ncandidates,init_guess, options);
D = (size(xtrain_norm,1)-1)/2;
new_x = new_x.*(max_x(1:D)-min_x(1:D)) + min_x(1:D);
end

function [I, dIdx]= adaptive_sampling_preference(x, theta, xtrain, ctrain, kernelfun, modeltype, post)
d = size(x,1);
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx] =  prediction_bin_preference(theta, xtrain, ctrain, x, kernelfun, 'modeltype', modeltype, 'post', post);

h = @(p) -p.*log(p+eps) - (1-p).*log(1-p+eps);

% for a gaussian cdf link function:
C = sqrt(pi*log(2)/2);

I1 = h(mu_c);
I2 =  log(2)*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
I = I1 - I2;

%for a sigmoid link
%C = sqrt(2*log(2));
%I = h(mu_c) - 2*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);

dhdp = @(p) -log(p)+log(1-p);
arg = mu_y./sqrt(sigma2_y+C^2);
dI1dx = (((dmuy_dx).*sqrt(sigma2_y+C^2)-mu_y.*dsigma2y_dx./(2*sqrt(sigma2_y+C^2)))./(sigma2_y+1)).*normpdf(arg).*dhdp(normcdf(arg));

dI2dx =I2.*(0.5*mu_y.^2.*dsigma2y_dx-mu_y.*(sigma2_y+C^2).*dmuy_dx)./((sigma2_y+C^2).^2)-I2./(2*(sigma2_y+C^2)).*dsigma2y_dx;
dI2dx = dI2dx(1:d,:);

dIdx = dI1dx - dI2dx;


I = -I;
dIdx = -dIdx;
end

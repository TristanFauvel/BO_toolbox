function [x_duel1, x_duel2,new_duel] = MaxEntChallenge(theta, xtrain_norm, ctrain, kernelfun, base_kernelfun, modeltype, max_x, min_x, lb_norm, ub_norm, condition, post, ~)
options.method = 'sd';
options.verbose = 1;
d = size(xtrain_norm,1)/2;
ncandidates =5;
init_guess = [];

x_best_norm = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, kernelfun, x0,modeltype, ost), lb_norm, ub_norm, ncandidates,init_guess, options);

x_duel1 =  x_best_norm.*(max_x(1:d)-min_x(1:d)) + min_x(1:d);

new = multistart_minConf(@(x)adaptive_sampling(theta, xtrain_norm, ctrain, x, x_best_norm, kernelfun, modeltype, post), lb_norm, ub_norm, ncandidates,init_guess, options);
x_duel2 = new.*(max_x(1:d)-min_x(1:d)) + min_x(1:d);

new_duel= [x_duel1; x_duel2];
end


function [I, dIdx] = adaptive_sampling(theta, xtrain_norm, ctrain, x, x_best_norm, kernelfun, modeltype, post)
d = size(xtrain_norm,1)/2;

xduel = [x;x_best_norm];
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc, dvar_muc_dx] =  prediction_bin_preference(theta, xtrain_norm, ctrain, xduel, kernelfun, 'post', post, 'modeltype', modeltype);

% h = @(p) -p.*log(p+eps) - (1-p).*log(1-p+eps);
h = @(p) -p.*log(p) - (1-p).*log(1-p);

% for a gaussian cdf link function: 
C = sqrt(pi*log(2)/2);

I1 = h(mu_c);
I2 =  log(2)*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
I = I1 - I2;
  
%for a sigmoid link
%C = sqrt(2*log(2));
%I = h(mu_c) - 2*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);

dhdp = @(p) -log(p)+log(1-p);
arg = mu_y./sqrt(sigma2_y+1);

dI1dx = (((dmuy_dx).*sqrt(sigma2_y+1)-mu_y.*dsigma2y_dx./(2*sqrt(sigma2_y+1)))./(sigma2_y+1)).*normpdf(arg).*dhdp(normcdf(arg));

dI2dx =I2.*(0.5*mu_y.^2.*dsigma2y_dx-mu_y.*(sigma2_y+C^2).*dmuy_dx)./((sigma2_y+C^2).^2)-I2./(2*(sigma2_y+C^2)).*dsigma2y_dx;
dI2dx = dI2dx;

dIdx = dI1dx - dI2dx;
 
I = -I;
dIdx = -dIdx(1:d,:);
end
%
% function [I, dI_dx] = adaptive_sampling(theta, xtrain_norm, ctrain, x, x_best_norm, kernelfun, modeltype)
% d = size(xtrain_norm,1)/2;
% 
% xduel = [x;x_best_norm];
% [mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc, dvar_muc_dx] =  prediction_bin_preference(theta, xtrain_norm, ctrain, xduel, kernelfun);
% 
% % h = @(p) -p.*log(p+eps) - (1-p).*log(1-p+eps);
% 
% 
% % for a gaussian cdf link function: 
% C = sqrt(pi*log(2)/2);
% I =log(2)*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
% % dI_dx =I.*(-4*mu_c*dmuc_dx*(sigma2_y+C)+mu_c^2*4*dsigma2y_dx)./(4*(sigma2_y+C).^3) -0.5*I/(sigma2_y+C^2).*dsigma2y_dx;
% dI_dx =I.*(0.5*mu_y.^2.*dsigma2y_dx-mu_y.*(sigma2_y+C^2).*dmuy_dx)./((sigma2_y+C^2).^2)-I./(2*(sigma2_y+C^2)).*dsigma2y_dx;
% 
% dI_dx =dI_dx;
% dI_dx = dI_dx(1:d,:);
% 
% I = -I;
% dI_dx = -dI_dx;
% %for a sigmoid link
% %C = sqrt(2*log(2));
% 
% %I = h(mu_c) - 2*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
%  
% end

% f = @(x) to_delete_adaptive_sampling(theta, xtrain_norm, ctrain, x, x_best_norm, kernelfun, modeltype)
% xrange = linspace(0,1,100);
% xrange = [xrange;xrange];
% test = zeros(d,100);
% result = zeros(d,100);
% 
% for i = 1:100
%     [a, result(:,i)] = to_delete_adaptive_sampling(theta, xtrain_norm, ctrain, xrange(:,i), x_best_norm, kernelfun, modeltype);
%     test(:,i)=  test_matrix_deriv(f, xrange(:,i), 1e-10);
% end
% figure();
% plot(result(:)); hold on;
% plot(test(:)); hold off;
% max((result(:) - test(:)).^2)
% 
% xduel = [xrange;x_best_norm.*ones(d,100)];
% [mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc, dvar_muc_dx] =  prediction_bin_preference(theta, xtrain_norm, ctrain, xduel, kernelfun);
% 
% figure();
% errorshaded(xrange, mu_y,sigma2_y);
% 
%  [I, dI_dx] = adaptive_sampling(theta, xtrain_norm, ctrain, xrange, x_best_norm.*ones(d,100),kernelfun, modeltype)
%  figure();
%  plot(-I);
function [U, dUdx] = knowledge_grad(theta, xtrain_norm, ctrain, xt, kernelfun, modeltype, post, c0, c1, xbest, ybest, lb_norm,ub_norm, link, x0)
ncandidates = 5;
init_guess = xbest;
options.verbose= 1;
options.method = 'lbfgs';
regularization = 'nugget';
[mu_c,  ~, ~, ~, dmuc_dx] =  prediction_bin(theta, xtrain_norm, ctrain, xt, kernelfun, modeltype, post, regularization, link);

post0 =  prediction_bin(theta, [xtrain_norm,xt], c0, [], kernelfun, modeltype, post, regularization);
post1 =  prediction_bin(theta, [xtrain_norm,xt], c1, [], kernelfun, modeltype, post, regularization);
if nargin == 14
    [xbest1, ybest1] = multistart_minConf(@(x)to_maximize_value_function(theta, [xtrain_norm, xt], c1, x, kernelfun, x0, modeltype, post1), lb_norm, ub_norm, ncandidates, init_guess, options);
    [xbest0, ybest0] = multistart_minConf(@(x)to_maximize_value_function(theta, [xtrain_norm, xt], c0, x, kernelfun, x0, modeltype, post0), lb_norm, ub_norm, ncandidates, init_guess, options);
    
else
    [xbest1, ybest1] = multistart_minConf(@(x)to_maximize_mean_bin_GP(theta, [xtrain_norm, xt], c1, x, kernelfun,modeltype, post1), lb_norm, ub_norm, ncandidates, init_guess, options);
    [xbest0, ybest0] = multistart_minConf(@(x)to_maximize_mean_bin_GP(theta, [xtrain_norm, xt], c0, x, kernelfun,modeltype, post0), lb_norm, ub_norm, ncandidates, init_guess, options);
    
end

ybest1 = - ybest1; ybest0 = -ybest0;

K = post0.K;
[k, ~, dkdx] = kernelfun(theta, [xtrain_norm, xt], xt, false, 'false');
dkdx = squeeze(dkdx);
n = size(K,1);
D = size(xt,1);
dKdx = zeros(n,n,D); %WARNING : modify for D ~= 1
dKdx(end,:,:) = dkdx;
dKdx(:,end,:) = dkdx;

dybest0dx = zeros(D,1);
dybest1dx = zeros(D,1);


for i = 1:D
    % Derivative case c = 0
    dystar0dx = (eye(n)+K*post0.D)\(dKdx(:,:,i)*post0.dloglike);
    dybest0dx(i) = dkdx(:,i)'*post0.dloglike - k'*post0.D.*dystar0dx;
    % Derivative case c = 1
    dystar1dx = (eye(n)+K*post1.D)\(dKdx(:,:,i)*post1.dloglike);
    dybest1dx(i)  = dkdx(:,i)'*post1.dloglike - k'*post1.D.*dystar1dx;
end
 
U = mu_c.*ybest1 + (1-mu_c).*ybest0 -ybest;

dmuc_dx = squeeze(dmuc_dx);
dUdx = dmuc_dx*(ybest1 -ybest0) + mu_c*dybest1dx + (1-mu_c)*dybest0dx;

U = -U;
dUdx = -dUdx;

end
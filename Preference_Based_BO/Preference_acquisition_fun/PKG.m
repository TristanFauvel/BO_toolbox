function [new_x, new_x_norm] = PKG(theta, xtrain_norm, ctrain, kernelfun, base_kernelfun, modeltype, max_x, min_x, lb_norm, ub_norm, condition, post, kernel_approx)

if ~strcmp(modeltype, 'laplace')
    error('This acquisition function is only implemented with Laplace approximation')
end
init_guess = [];
options.method = 'lbfgs';
options.verbose = 1;
ncandidates = 5;
[xbest, ybest] = multistart_minConf(@(x)to_maximize_value_GP(theta, xtrain_norm, ctrain, x, kernelfun, condition.x0, modeltype, post), lb_norm, ub_norm, ncandidates, init_guess, options);
ybest = -ybest;

c0 = [ctrain, 0];
c1 = [ctrain,1];

new_x_norm  = multistart_minConf(@(x)knowledge_grad(theta, xtrain_norm, ctrain, x, kernelfun, modeltype, post, c0, c1, xbest, ybest, condition), lb_norm, ub_norm, ncandidates, init_guess, options);

new_x = new_x_norm.*(max_x-min_x) + min_x;

end

function [U, dUdx] = knowledge_grad(theta, xtrain_norm, ctrain, xt, kernelfun, modeltype, post, c0, c1, xbest, ybest, condition)
ncandidates = 2;
init_guess = xbest;
options.verbose= 1;
options.method = 'lbfgs';

[mu_c,  ~, ~, ~, dmuc_dx] =  prediction_bin(theta, xtrain_norm, ctrain, xt, kernelfun, modeltype, post, regularization);

post0 =  prediction_bin(theta, [xtrain_norm,xt], c0, [], kernelfun, modeltype, post, regularization);
post1 =  prediction_bin(theta, [xtrain_norm,xt], c1, [], kernelfun, modeltype, post, regularization);

[xbest1, ybest1] = multistart_minConf(@(x)to_maximize_mean_bin_GP(theta, [xtrain_norm, xt], c1, x, kernelfun, condition.x0, modeltype, post1), lb_norm, ub_norm, ncandidates, init_guess, options);
[xbest0, ybest0] = multistart_minConf(@(x)to_maximize_mean_bin_GP(theta, [xtrain_norm, xt], c0, x, kernelfun, condition.x0, modeltype, post0), lb_norm, ub_norm, ncandidates, init_guess, options);
ybest1 = - ybest1; ybest0 = -ybest0;

K = post0.K;
[k, ~, dkdx] = kernelfun(theta, [xtrain_norm,xt], xt, false, 'false');
dkdx = squeeze(dkdx);
n = size(K,1);
D = size(xt,1);
dKdx = zeros(n,n,D); %WARNING : modify for D ~= 1
dKdx(end,:,:) = dkdx;
dKdx(:,end,:) = dkdx;

dlogistic = @(x) logistic(x).*(1-logistic(x));
D = size(xt,1);
dybest0dx = zeros(D,1);
dybest1dx = zeros(D,1);

for i = 1:D
% Derivative case c = 0
dystar0dx = (eye(n)+K*D)\(dKdx(:,:,i)*(c0(:)-logistic(post0.ystar)));
dybest0dx(i) = dkdx(:,i)'*(c0(:)-logistic(post0.ystar))-k'*(dlogistic(post0.ystar).*dystar0dx);
% Derivative case c = 1
dystar1dx = (eye(n)+K*D)\(dKdx(:,:,i)*(c1(:)-logistic(post1.ystar)));
dybest1dx(i)  = dkdx(:,i)'*(c1(:)-logistic(post1.ystar))-k'*(dlogistic(post1.ystar).*dystar1dx);
end
       
U = mu_c.*ybest1 + (1-mu_c).*ybest0 -ybest;

dmuc_dx = squeeze(dmuc_dx);
dUdx = dmuc_dx*(ybest1 -ybest0) + mu_c*dybest1dx + (1-mu_c)*dybest0dx; 

U = -U;
dUdx = -dUdx;

end
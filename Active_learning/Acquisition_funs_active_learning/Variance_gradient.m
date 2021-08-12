function [new_x, new_x_norm] = Variance_gradient(theta, xtrain_norm, ctrain, kernelfun,modeltype, max_x, min_x, lb_norm, ub_norm, post)
if ~strcmp(modeltype, 'laplace')
    error('This acquisition function is only implemented with Laplace approximation')
end
ncandidates = 10;
init_guess = [];
options.method = 'lbfgs';
options.verbose = 1;
new_x_norm = multistart_minConf(@(x)adaptive_sampling_binary(x, theta, xtrain_norm, ctrain,model, post), model.lb_norm, model.ub_norm, ncandidates,init_guess, options);
new_x = new_x_norm.*(model.max_x-model.min_x) + model.min_x;
end

function [vargrad_x, dvargrad_x_dx] = vargrad(theta, xtrain_norm, ctrain, x_duel1, x, model, post)
ncandidates = 5;
init_guess = xbest;
options.verbose= 1;
options.method = 'lbfgs';

[mu_c,  ~, ~, ~, dmuc_dx,~,~,~, var_muc, dvar_muc_dx] =  prediction_bin(theta, xtrain_norm, ctrain, x, model, post);

var_muc = -var_muc;
dvar_muc_dx = -dvar_muc_dx;

c0 = [ctrain, 0];
c1 = [ctrain,1];


post0 =  prediction_bin(theta, [xtrain_norm,xt], c0, [], model, post);
post1 =  prediction_bin(theta, [xtrain_norm,xt], c1, [], model, post);

[xmaxvar1, maxvar1] = multistart_minConf(@(x)to_maximize_var_bin_GP(theta, [xtrain_norm, xt], c1, x, model, post1), lb_norm, ub_norm, ncandidates, init_guess, options);
[xmaxvar0, maxvar0] = multistart_minConf(@(x)to_maximize_var_bin_GP(theta, [xtrain_norm, xt], c0, x, model, post0), lb_norm, ub_norm, ncandidates, init_guess, options);
maxvar0 = - maxvar0; maxvar1 = -maxvar1;

K = kernelfun(theta, [xtrain_norm, xt],[xtrain_norm, xt]);
[k, ~, dkdx] = kernelfun(theta, [xtrain_norm, xt], xt);
dkdx = squeeze(dkdx);
n = size(K,1);
D = size(xt,1);
dKdx = zeros(n,n,D); %WARNING : modify for D ~= 1
dKdx(end,:,:) = dkdx;
dKdx(:,end,:) = dkdx;

dlogistic = @(x) logistic(x).*(1-logistic(x));
D = size(xt,1);
dmaxvar0dx = zeros(D,1);
dmaxvar1dx = zeros(D,1);

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


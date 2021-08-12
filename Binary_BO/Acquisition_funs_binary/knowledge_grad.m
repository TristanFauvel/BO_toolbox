function [U, dUdx] = knowledge_grad(theta, xtrain_norm, ctrain, xt,model, post, c0, c1, xbest, ybest, lb_norm,ub_norm)

kernelfun = model.kernelfun;
modeltype = model.modeltype;

ncandidates = 2;
init_guess = [];
options.verbose= 1;
options.method = 'lbfgs';
[mu_c,  ~, ~, ~, dmuc_dx] =  prediction_bin(theta, xtrain_norm, ctrain, xt, model, post);

post0 =  prediction_bin(theta, [xtrain_norm,xt], c0, [], model, post);
post1 =  prediction_bin(theta, [xtrain_norm,xt], c1, [], model, post);
if strcmp(model.type, 'preference')
    [xbest1, ybest1] = multistart_minConf(@(x)to_maximize_value_function(theta, [xtrain_norm, xt], c1, x, model, post1), lb_norm, ub_norm, ncandidates, init_guess, options);
    [xbest0, ybest0] = multistart_minConf(@(x)to_maximize_value_function(theta, [xtrain_norm, xt], c0, x, model, post0), lb_norm, ub_norm, ncandidates, init_guess, options);
    
else
    [xbest1, ybest1] = multistart_minConf(@(x)to_maximize_mean_bin_GP(theta, [xtrain_norm, xt], c1, x, model, post1), lb_norm, ub_norm, ncandidates, init_guess, options);
    [xbest0, ybest0] = multistart_minConf(@(x)to_maximize_mean_bin_GP(theta, [xtrain_norm, xt], c0, x, model, post0), lb_norm, ub_norm, ncandidates, init_guess, options);
    
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
    dybest0dx(i) = dkdx(:,i)'*post0.dloglike - k'*post0.D*dystar0dx;
    % Derivative case c = 1
    dystar1dx = (eye(n)+K*post1.D)\(dKdx(:,:,i)*post1.dloglike);
    dybest1dx(i)  = dkdx(:,i)'*post1.dloglike - k'*post1.D*dystar1dx;
end
 
U = mu_c.*ybest1 + (1-mu_c).*ybest0 -ybest;

dmuc_dx = squeeze(dmuc_dx);
dUdx = dmuc_dx*(ybest1 -ybest0) + mu_c*dybest1dx + (1-mu_c)*dybest0dx;

U = -U;
dUdx = -dUdx;

end
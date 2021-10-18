function [U, dUdx] = knowledge_grad(theta, xtrain_norm, ctrain, xt,model, post, c0, c1, xbest, ybest, lb_norm,ub_norm)

kernelfun = model.kernelfun;

ncandidates =10;
init_guess = [];
options.verbose= 1;
options.method = 'lbfgs';
[mu_c,  ~, ~, ~, dmuc_dx] =  model.prediction(theta, xtrain_norm, ctrain, xt, post);

post0 =  model.prediction(theta, [xtrain_norm,xt], c0, [], model, []);
post1 =  model.prediction(theta, [xtrain_norm,xt], c1, [], model, []);
if strcmp(model.type, 'preference')
    [xbest1, ybest1] = multistart_minConf(@(x)to_maximize_value_function(theta, [xtrain_norm, xt], c1, x, model, post1), lb_norm, ub_norm, ncandidates, init_guess, options);
    [xbest0, ybest0] = multistart_minConf(@(x)to_maximize_value_function(theta, [xtrain_norm, xt], c0, x, model, post0), lb_norm, ub_norm, ncandidates, init_guess, options);
    xbest1 = [xbest1; model.condition.x0];
    xbest0 = [xbest0; model.condition.x0];
else
    init_guess = xbest;
    [xbest1, ybest1] = multistart_minConf(@(x)to_maximize_mean_bin_GP(theta, [xtrain_norm, xt], c1, x, model, post1), lb_norm, ub_norm, ncandidates, init_guess, options);
    [xbest0, ybest0] = multistart_minConf(@(x)to_maximize_mean_bin_GP(theta, [xtrain_norm, xt], c0, x, model, post0), lb_norm, ub_norm, ncandidates, init_guess, options);
    
end

ybest1 = - ybest1;
ybest0 = -ybest0;

U = mu_c.*ybest1 + (1-mu_c).*ybest0 -ybest;
U = -U;

if nargout >1
    K = post0.K;
    
    %%
    [k1, ~, dk1dx] = kernelfun(theta, xbest1, [xtrain_norm, xt], false, model.regularization);
    [k0, ~, dk0dx] = kernelfun(theta, xbest0, [xtrain_norm, xt], false, model.regularization);
    k1= k1';
    k0 = k0';
    D = size(xtrain_norm,1);
    
    if D>1
        dk1dx = squeeze(dk1dx(:,:,end,:));
        dk0dx = squeeze(dk0dx(:,:,end,:));
    else
        dk1dx = dk1dx(:,:,end)';
        dk0dx = dk0dx(:,:,end)';
    end
    %%
    
    [k, ~, dkdx] = kernelfun(theta, [xtrain_norm, xt], xt, false, model.regularization);
    dkdx = squeeze(dkdx); % (ntr+1) x D
    if isfield(model, 'context') && model.context ==1
        [~, ~, ~, dkxxdx] = kernelfun(theta, xt, xt, false, model.regularization);
        dkdx(end,:)= dkxxdx; % dkdx(end,1)= dkxxdx;
    end
    
    n = size(K,1);
    dKdx = zeros(n,n,D);
    dKdx(end,:,:) = dkdx;
    dKdx(:,end,:) = dkdx;
    
    dybest0dx = zeros(D,1);
    dybest1dx = zeros(D,1);
    
    if strcmp(model.modeltype, 'laplace')
        for i = 1:D
            % Derivative case c = 0
            dystar0dx = (eye(n)+K*post0.D)\(dKdx(:,:,i)*post0.dloglike);
            dybest0dx(i) = dk0dx(:,i)'*post0.dloglike - k0'*(diag(post0.D).*dystar0dx);
            % Derivative case c = 1
            dystar1dx = (eye(n)+K*post1.D)\(dKdx(:,:,i)*post1.dloglike);
            dybest1dx(i) = dk1dx(:,i)'*post1.dloglike - k1'*(diag(post1.D).*dystar1dx);
        end
    else
        error('This acquisition function is only implemented with Laplace approximation')
    end
    
    dmuc_dx = squeeze(dmuc_dx);
    dUdx = dmuc_dx*(ybest1 -ybest0) + mu_c*dybest1dx + (1-mu_c)*dybest0dx;
    dUdx = -dUdx;
end
end
function [new_x,new_x_norm, L] = UCB_binary(theta, xtrain_norm, ctrain, model, post, approximation, varargin)
opts = namevaluepairtostruct(struct( ...
    'task', 'max',...
    'e', norminv(0.99) ...
    ), varargin);

UNPACK_STRUCT(opts, false)
options.method = 'lbfgs';

ncandidates= 10;
init_guess = [];
[new_x_norm, L] = multistart_minConf(@(x)ucb(theta, xtrain_norm, x, ctrain, model, post, task, e), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);
L = -L;
new_x = new_x_norm.*(model.ub-model.lb) + model.lb;

end

% x = linspace(0,1,1000);
% for i = 1:1000
%     [ucb_val(i), ducb_dx]= ucb(theta, xtrain_norm, x(i), ctrain, model, post, task, e);
% end
% figure();
% plot(x, ucb_val)

function [ucb_val, ducb_dx]= ucb(theta, xtrain_norm, x, ctrain, model, post, task, e)

if strcmp(task, 'max')
    [mu_c, sigma_c, dmuc_dx, dsigma_c_dx] =  gpbin(theta, xtrain_norm, ctrain, x, model, post);
elseif strcmp(task, 'average')
    mu_c = integral(@(s) fmu_c(theta, xtrain_norm, ctrain, [s;x*ones(1,size(s,2))], model, post), zeros(1,model.ns), ones(1,model.ns), 'ArrayValued', true);
    sigma_c = integral(@(s) fsigma_c(theta, xtrain_norm, ctrain, [s;x*ones(1,size(s,2))], model, post), zeros(1,model.ns), ones(1,model.ns), 'ArrayValued', true);
    dmuc_dx = integral(@(s) fdmuc_dx(theta, xtrain_norm, ctrain, [s;x*ones(1,size(s,2))], model, post), zeros(1,model.ns), ones(1,model.ns), 'ArrayValued', true);
    dsigma_c_dx = integral(@(s) fdsigma_c_dx(theta, xtrain_norm, ctrain, [s;x*ones(1,size(s,2))], model, post), zeros(1,model.ns), ones(1,model.ns), 'ArrayValued', true);
end
ucb_val = mu_c + e*sigma_c;
ucb_val = -ucb_val;
ducb_dx = -(dmuc_dx(:) + e*dsigma_c_dx(:));
end

function mu_c = fmu_c(theta, xtrain_norm, ctrain, x, model, post)
mu_c  =  prediction_bin(theta, xtrain_norm, ctrain, x, model, post);
end

function sigma_c =  fsigma_c(theta, xtrain_norm, ctrain, x, model, post)
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] =  prediction_bin(theta, xtrain_norm, ctrain, x, model, post);
sigma_c = sqrt(var_muc);
end

function dmuc_dx = fdmuc_dx(theta, xtrain_norm, ctrain, x, model, post)
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx] =  prediction_bin(theta, xtrain_norm, ctrain, x, model, post);
dmuc_dx = dmuc_dx((model.ns+1):end);
end

function dsigma_c_dx  =  fdsigma_c_dx(theta, xtrain_norm, ctrain, x, model, post)
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc, dvar_muc_dx] =  prediction_bin(theta, xtrain_norm, ctrain, x, model, post);
sigma_c = sqrt(var_muc);
dsigma_c_dx = dvar_muc_dx./(2*sigma_c);
dsigma_c_dx = dsigma_c_dx((model.ns+1):end);
end

function [mu_c, sigma_c, dmuc_dx, dsigma_c_dx]  = gpbin(theta, xtrain_norm, ctrain, x, model, post)
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc, dvar_muc_dx] =  prediction_bin(theta, xtrain_norm, ctrain, x, model, post);
sigma_c = sqrt(var_muc);
dsigma_c_dx = dvar_muc_dx./(2*sigma_c);
% dsigma_c_dx = dsigma_c_dx((model.ns+1):end);
end
function [sample_normalized, sample] = sample_max_binary_GP(approximation, xtrain_norm, ctrain, theta, model, post, s0)
 
options.method = 'lbfgs';
options.verbose = 1;
ncandidates= 5;

[sample_g, dsample_g_dx] = sample_binary_GP_precomputed_features(xtrain_norm, ctrain, theta,model, approximation, post);

init_guess = [];
    xdims =  (model.ns+1):model.D;

if strcmp(model.task, 'max')
    f = sample_g;
    dfdx = dsample_g_dx;
    xdims = (model.ns+1):model.D;
elseif strcmp(model.task, 'average')
    tol = 1e-2;
    sdims = 1:model.ns;
    xdims =  (model.ns+1):model.D;
    f = @(x) integral(@(s) sample_g([s;x*ones(1,size(s,2))]), model.lb_norm(sdims), model.ub_norm(sdims));
    dfdx = @(x) integral(@(s) dsample_g_dx([s;x*ones(1,size(s,2))]), model.lb_norm(sdims), model.ub_norm(sdims),'ArrayValued', true,'RelTol', tol);
end
sample_normalized = multistart_minConf(@(x)takemax(x,f, dfdx, xdims), model.lb_norm(xdims), model.ub_norm(xdims), ncandidates,init_guess, options);
sample = sample_normalized.*(model.ub(xdims)-model.lb(xdims)) + model.lb(xdims);

end

function [fx, dfdx] = takemax(x,f,df, xdims)
if any(isnan(x))
    warning('x is NaN')
end
% Function that groups f and df to use minFunc
fx = -f(x);
dfdx = - df(x);
dfdx = dfdx(xdims);
dfdx = dfdx(:);
end


% xx = linspace(0,1,1000);
% for i = 1:1000
%     vf(i) = f(xx(i));
% end
% figure()
% plot(xx, vf)
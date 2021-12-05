function [sample_normalized, sample] = sample_max_binary_GP(approximation, xtrain_norm, ctrain, theta, model, post, optim)

options.method = 'lbfgs';
options.verbose = 1;
ncandidates= 5;

[sample_g, dsample_g_dx] = sample_binary_GP_precomputed_features(xtrain_norm, ctrain, theta,model, approximation, post);

init_guess = [];

if strcmp(optim.task, 'max')
    f = sample_g;
    dfdx = dsample_g_dx;    
    sample_normalized = multistart_minConf(@(x)takemax(x,f, dfdx, model.sdims), model.lb_norm, model.ub_norm, ncandidates,init_guess, options);
elseif strcmp(optim.task, 'average')
    if optim.ongrid
        f = sample_g(model.grid_norm);
        f_x = mean(f,model.sdims);
        [fmax, id] =  max(f_x);
        sample_normalized = model.grid(model.xdims, id);
    else
        error('Not implemented')
    end
end
sample = sample_normalized.*(model.ub(model.xdims)-model.lb(model.xdims)) + model.lb(model.xdims);

% elseif strcmp(optim.task, 'average')
%     tol = 1e-2;
%     sdims = 1:model.ns;
%     xdims =  (model.ns+1):model.D;
%     f = @(x) integral(@(s) sample_g([s;x*ones(1,size(s,2))]), model.lb_norm(sdims), model.ub_norm(sdims));
%     dfdx = @(x) integral(@(s) dsample_g_dx([s;x*ones(1,size(s,2))]), model.lb_norm(sdims), model.ub_norm(sdims),'ArrayValued', true,'RelTol', tol);
% end
end

function [fx, dfdx] = takemax(x,f,df, sdims)
if any(isnan(x))
    warning('x is NaN')
end
% Function that groups f and df to use minFunc
fx = -f(x);
dfdx = - df(x);
%dfdx = dfdx(xdims);
 dfdx(sdims) = 0;
 dfdx = dfdx(:);

end



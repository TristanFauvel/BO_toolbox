function [new_x, new_x_norm, ei] = EI_Tesch(theta, xtrain_norm, ctrain,model, post, approximation, optim)
%Expected improvement criterion by Tesch et al 2013
if ~strcmp(func2str(model.link), 'normcdf')
    error('Function only implemented for a normcdf link')
end

init_guess = [];
options.method = 'lbfgs';
options.verbose = 1;
ncandidates = optim.AF_ncandidates;
[xbest, mu_c_best] = optimize_AF(@(x)to_maximize_mu_c(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm,  model.ub_norm, ncandidates, init_guess, options);
ybest = norminv(mu_c_best);

nsamps= 1e5;
e = randn(1, nsamps);


[new_x_norm,ei] = optimize_AF(@(x)ExpImp(theta, xtrain_norm, ctrain, x, model, post,mu_c_best, ybest, e), model.lb_norm,  model.ub_norm, ncandidates, init_guess, options);
new_x = new_x_norm.*(model.ub-model.lb) + model.lb;
end

function [mu_c,  dmuc_dx] = to_maximize_mu_c(theta, xtrain_norm, ctrain, x,model, post)
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx] =  model.prediction(theta, xtrain_norm, ctrain, x, post);
dmuc_dx= squeeze(dmuc_dx);
end

function [ei,deidx] = ExpImp(theta, xtrain_norm, ctrain, x, model, post, mu_c_best, ybest,e)
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx] =  model.prediction(theta, xtrain_norm, ctrain, x, post);

sigma_y = sqrt(sigma2_y);

samples = mu_y + sigma_y*e;
samples(ybest>samples)=0;
arg = model.link(samples) - mu_c_best;
ei = mean(arg);

[g, dgdx, dgdmu, dgdsigma] =  Gaussian_fun(samples, mu_y, sigma_y);

dsigmay_dx= dsigma2y_dx./(2*sigma_y);
deidx = mean(arg.*(dgdmu.*dmuy_dx + dgdsigma.*dsigmay_dx)./g,2);
end

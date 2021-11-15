function [new_x, new_x_norm] = EI_Tesch(theta, xtrain_norm, ctrain,model, post, approximation, optimization)
%Expected improvement criterion by Tesch et al 2013
if ~strcmp(func2str(model.link), 'normcdf')
    error('Function only implemented for a normcdf link')
end

init_guess = [];
options.method = 'sd';
options.verbose = 1;
ncandidates = 10;
[xbest, mu_c_best] = multistart_minConf(@(x)to_maximize_mu_c(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm,  model.ub_norm, ncandidates, init_guess, options);
mu_c_best = - mu_c_best;
ybest = norminv(mu_c_best);
[new_x_norm,ei] = multistart_minConf(@(x)ExpImp(theta, xtrain_norm, ctrain, x, model, post,mu_c_best, ybest), model.lb_norm,  model.ub_norm, ncandidates, init_guess, options);
new_x = new_x_norm.*(model.ub-model.lb) + model.lb;
end

function [mu_c,  dmuc_dx] = to_maximize_mu_c(theta, xtrain_norm, ctrain, x,model, post)
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx] =  model.prediction(theta, xtrain_norm, ctrain, x, post);
mu_c = -mu_c;
dmuc_dx= -squeeze(dmuc_dx);
end


function [ei,deidx] = ExpImp(theta, xtrain_norm, ctrain, x, model, post, mu_c_best, ybest)
nsamps= 1e5;
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx] =  model.prediction(theta, xtrain_norm, ctrain, x, post);

sigma_y = sqrt(sigma2_y);

pd = makedist('Normal');
pd.mu = mu_y;
pd.sigma = sigma_y;
t = truncate(pd,ybest,inf);
samples= random(t,1,nsamps);

arg = model.link(samples) - mu_c_best;
ei = mean(arg);

[g, dgdx, dgdmu, dgdsigma] =  Gaussian(samples, mu_y, sigma_y);

dsigmay_dx= dsigma2y_dx./(2*sigma_y);
deidx = mean(arg.*(dgdmu.*dmuy_dx + dgdsigma.*dsigmay_dx)./g,2);

ei = -ei;
deidx = -deidx;
end

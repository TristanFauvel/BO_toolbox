function [new_x,new_x_norm, u] = UCB_binary_latent(theta, xtrain_norm, ctrain,model, post, approximation, optimization, varargin)

opts = namevaluepairtostruct(struct('e', 1), varargin); % 1 is used in the original paper by Tesch et al (2013)
   

UNPACK_STRUCT(opts, false)

options.method = 'lbfgs';

ncandidates= optimization.AF_ncandidates;
init_guess = [];
 
[new_x_norm, u] =  optimize_AF(@(x)ucb(theta, xtrain_norm, x, ctrain, model, post, e), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);

new_x = new_x_norm.*(model.ub-model.lb) + model.lb;

end


function [ucb_val, ducb_dx]= ucb(theta, xtrain_norm, x, ctrain, model, post, e)
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx] =  model.prediction(theta, xtrain_norm, ctrain, x, post);
sigma_y = sqrt(sigma2_y);
dsigma_y_dx = dsigma2y_dx./(2*sigma_y);
 ucb_val = mu_y + e*sigma_y;ucb_val = ucb_val;
ducb_dx = (dmuy_dx(:) + e*dsigma_y_dx(:));
end


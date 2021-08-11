function [new_x, new_x_norm]= Thompson_sampling(theta, xtrain_norm, ytrain, meanfun, kernelfun, kernelname, max_x, min_x, lb_norm, ub_norm, approximation)

decoupled_bases= 1;
[new_x, new_x_norm] = sample_max_GP(approximation, xtrain_norm, ytrain, theta,kernelfun, kernelname, decoupled_bases, max_x, min_x, lb_norm, ub_norm);

return




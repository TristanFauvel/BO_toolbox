function [new_x,new_x_norm] = TS_binary(theta, xtrain_norm, ctrain, kernelfun, modeltype, max_x, min_x, lb_norm, ub_norm, post, kernel_approx)

decoupled_bases = 1;
[new_x_norm, new_x] = sample_max_binary_GP(kernel_approx, xtrain_norm, ctrain, theta,kernelfun, decoupled_bases, modeltype,  post,max_x, min_x, lb_norm, ub_norm);


return




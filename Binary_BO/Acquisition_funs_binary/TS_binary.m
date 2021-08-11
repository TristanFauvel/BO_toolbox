function [new_x,new_x_norm] = TS_binary(theta, xtrain_norm, ctrain,model, max_x, min_x, lb_norm, ub_norm, post, approximation)

decoupled_bases = 1;
[new_x_norm, new_x] = sample_max_binary_GP(approximation, xtrain_norm, ctrain, theta,kernelfun, decoupled_bases, modeltype,  post,max_x, min_x, lb_norm, ub_norm);


return




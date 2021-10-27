function [new_x,new_x_norm] = TS_binary(theta, xtrain_norm, ctrain,model, post, approximation)


[new_x_norm, new_x] = sample_max_binary_GP(approximation, xtrain_norm, ctrain, theta,model, post);
return




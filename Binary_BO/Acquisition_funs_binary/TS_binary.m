function [new_x,new_x_norm, fmax] = TS_binary(theta, xtrain_norm, ctrain,model, post, approximation, optim)


[new_x_norm, new_x, fmax] = sample_max_GP(approximation, xtrain_norm, ctrain, theta,model, post, optim);
return



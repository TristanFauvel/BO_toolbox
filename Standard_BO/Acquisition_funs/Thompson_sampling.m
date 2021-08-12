function [new_x, new_x_norm]= Thompson_sampling(theta, xtrain_norm, ytrain, model, post, approximation)

[new_x, new_x_norm] = sample_max_GP(approximation, xtrain_norm, ytrain, theta,model);

return




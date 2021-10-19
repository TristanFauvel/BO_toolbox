function [new_x, new_x_norm] = random_acquisition(theta, xtrain_norm, ytrain, model, post, approximation, optimization)

D = numel(max_x);
new_x_norm = rand_interval(lb_norm,ub_norm);
new_x = new_x_norm.*(model.ub-model.lb) + model.lb;


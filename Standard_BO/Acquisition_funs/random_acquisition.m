function [new_x, new_x_norm] = random_acquisition(theta, xtrain_norm, ytrain, meanfun, kernelfun, kernelname, max_x, min_x, lb_norm, ub_norm, kernel_approx)       

D = numel(max_x);
new_x_norm = rand_interval(lb_norm,ub_norm);
new_x = new_x_norm.*(max_x(1:D)-min_x(1:D)) + min_x(1:D);


function [new_x, new_x_norm]= random_acquisition_binary(theta, xtrain_norm, ctrain, model, post, approximation)
new_x_norm = rand_interval(model.lb_norm,model.ub_norm);
new_x = new_x_norm.*(model.ub-model.lb) + model.lb;


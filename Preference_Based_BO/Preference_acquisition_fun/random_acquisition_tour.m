function [new_x, new_x_norm] = random_acquisition_tour(theta, xtrain_norm, ctrain, model, post, approximation, optim)

new_x_norm = rand_interval(model.lb_norm,model.ub_norm,'nsamples', optim.batch_size);

new_x = new_x_norm.*(model.ub-model.lb) + model.lb;


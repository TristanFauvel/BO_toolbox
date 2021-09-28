function x = random_acquisition_tour(theta, xtrain_norm, ctrain, model, post, approximation)

xnorm= rand_interval(model.lb_norm,model.ub_norm,'nsamples', model.nsamples);


x = xnorm.*(model.max_x-model.min_x) + model.min_x;


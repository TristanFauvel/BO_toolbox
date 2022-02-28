function [new_x, new_x_norm, u] = BKG((theta, xtrain_norm, ctrain,model, post, approximation)
if ~strcmp(model.modeltype, 'laplace')
    error('This acquisition function is only implemented with Laplace approximation')
end

[xbest, ybest] =  model.maxmean(theta, xtrain_norm, ctrain, post);

c0 = [ctrain(:)', 0];
c1 = [ctrain(:)',1];

[new_x_norm, u] = optimize_AF(@(x)knowledge_grad(theta, xtrain_norm, ctrain, x,model, post, c0, c1, xbest, ybest,model.lb_norm, model.ub_norm), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);

new_x = new_x_norm.*(model.ub-model.lb) + model.lb;
end

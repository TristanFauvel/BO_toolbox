function new_x = kernelselfsparring_tour(theta, xtrain_norm, ctrain, modelmodeltype, max_x, min_x, lb_norm, ub_norm, condition, post, approximation, nsamples)

decoupled_bases = 1;
D = numel(lb_norm);
xnorm = zeros(D,nsamples);
for k =1:nsamples %sample g* from p(g*|D)
        [xnorm(:,k), new] = sample_max_preference_GP(approximation, xtrain_norm, ctrain, theta, model, decoupled_bases, post);
end
new_x = xnorm.*(max_x-min_x) + min_x;

end


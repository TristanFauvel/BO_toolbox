function new_x = kernelselfsparring_tour(theta, xtrain_norm, ctrain, kernelfun, base_kernelfun,modeltype, max_x, min_x, lb_norm, ub_norm, condition, post, kernel_approx, nsamples)

decoupled_bases = 1;
D = numel(lb_norm);
xnorm = zeros(D,nsamples);
for k =1:nsamples %sample g* from p(g*|D)
        [xnorm(:,k), new] = sample_max_preference_GP(kernel_approx, xtrain_norm, ctrain, theta,kernelfun, decoupled_bases, modeltype, base_kernelfun, post, condition, max_x, min_x, lb_norm, ub_norm);
end
new_x = xnorm.*(max_x-min_x) + min_x;

end


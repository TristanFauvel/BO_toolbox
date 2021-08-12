function new_x = kernelselfsparring_tour(theta, xtrain_norm, ctrain, model, post, approximation, nsamples)

D = model.D;
xnorm = zeros(D,nsamples);
for k =1:model.nsamples %sample g* from p(g*|D)
        [xnorm(:,k), new] = sample_max_preference_GP(approximation, xtrain_norm, ctrain, theta, model, post);
end
new_x = xnorm.*(max_x-min_x) + min_x;

end


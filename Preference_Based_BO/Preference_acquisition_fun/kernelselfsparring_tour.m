function [new_x, new_x_norm]= kernelselfsparring_tour(theta, xtrain_norm, ctrain, model, post, approximation, optim)

D = model.D;
new_x_norm = zeros(D, optim.batch_size);
new_x = zeros(D, optim.batch_size);
for k =1:optim.batch_size %sample g* from p(g*|D)
        [new_x_norm(:,k), new_x(:,k)] = model.sample_max_GP(approximation, xtrain_norm, ctrain, theta, post);     
end

end


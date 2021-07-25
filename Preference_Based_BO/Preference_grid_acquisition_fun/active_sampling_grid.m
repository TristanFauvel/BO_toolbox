function new_duel = active_sampling_grid(x, theta, xtrain, ctrain, kernelfun, xduels, modeltype, m, kernelname, post)

if m > 2 
error('Active sampling only possible with duels, not tournaments')
end
x0 = x(:,1);
[mu_c,  mu_y, sigma2_y] = prediction_bin(theta, xtrain, ctrain, [x; x0*ones(1,ntest^d)], kernelfun,kernelname, modeltype, post, regularization);

[mu_c_acq,  mu_y_acq, sigma2_y_acq] = prediction_bin(theta, xtrain, ctrain, xduels, kernelfun, kernelname,modeltype, post, regularization);
acq = adaptive_sampling(mu_c_acq, mu_y_acq, sigma2_y_acq);
new_id= find(acq==max(acq));
if numel(new_id)~=1
    new_id = randsample(new_id,1);    
end
new_duel= xduels(:,new_id);


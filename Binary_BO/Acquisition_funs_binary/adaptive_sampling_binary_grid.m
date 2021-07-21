function [new_x, idx, L] = adaptive_sampling_binary_grid(x, theta, xtrain, ctrain, kernelfun, modeltype)

[mu_c, mu_y, sigma2_y] =   prediction_bin(theta, xtrain, ctrain, x, kernelfun, 'modeltype', modeltype);

C = sqrt(pi*log(2)/2);

h = @(p) -p.*log(p+eps) - (1-p).*log(1-p+eps);

L = h(mu_c) - 0.6932*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);

idx= find(L==max(L));
if numel(idx)~=1
    idx = randsample(idx,1);    
end
nd = size(x,1);
if nd==1
    new_x = x(idx);
else
    new_x = x(:,idx);    
end
return

figure()
subplot(3,1,1)
plot(x,mu_c); hold on;
scatter(xtrain, ctrain)
subplot(3,1,2)
errorshaded(x,mu_y,sqrt(sigma2_y))
subplot(3,1,3)
plot(x,L)

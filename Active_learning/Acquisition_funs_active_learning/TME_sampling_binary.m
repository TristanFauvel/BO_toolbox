function  [new_x, new_x_norm, idx, L] = TME_sampling_binary(x, theta, xtrain_norm, ctrain,model, post)
xog = x;
ngrid = 300;
if size(x,2)>ngrid
    keep = randsample(size(x,2), ngrid);
    x = x(:,keep);
end
xnorm = (x - model.lb)./(model.ub - model.lb);

% gaussian_entropy = @(Sigma) 0.5+log(det(Sigma)) + 0.5*size(Sigma,1)*log(2*pi*exp(1));
gaussian_entropy = @(sigma2) 0.5*log(2*pi*exp(1)*sigma2);
regularization = 'nugget';
[mu_c,  mu_y, sigma2_y] =  prediction_bin(theta, xtrain_norm, ctrain, xnorm, model, post);
H1 = sum(gaussian_entropy(sigma2_y));

%% Expected total marginal entropy after observing y
c0 = [ctrain,0];
c1 = [ctrain,1];
H2 = zeros(size(x,2),1);
for i =1:size(x,2)
    %case y = 0
    [~, ~, sigma2_y0] =  prediction_bin(theta, [xtrain_norm, x(:,i)], c0, xnorm, model, post);
    %case y = 1
    [~,~, sigma2_y1] =  prediction_bin(theta, [xtrain_norm, x(:,i)], c1, xnorm, model, post);
    H20 = sum(gaussian_entropy(sigma2_y0));
    H21 = sum(gaussian_entropy(sigma2_y1));
    H2(i) = mu_c(i)*H21 + (1-mu_c(i))*H20;
end
L = H1-H2;

idx= find(L==max(L));
if numel(idx)~=1
    idx = randsample(idx,1);
end
new_x = x(:,idx);
new_x_norm = xnorm(:,idx);
idx = find(ismember(xog',new_x', 'rows'));

return


function [new_x, new_x_norm, idx, L] = MI_gradient_grid(x, theta, xtrain_norm, ctrain,model, post)
xog = x;
ngrid = 300;
if size(x,2)>ngrid
    keep = randsample(size(x,2), ngrid);
    x = x(:,keep);
end

xnorm = (x - model.lb)./(model.ub - model.lb);

mu_c =   prediction_bin(theta, xtrain_norm, ctrain, xnorm, model, post);

n = size(x,2);
L = zeros(1, n);
bald = BALD(theta, xtrain_norm, ctrain, xnorm, model, post);
for i = 1:n
    L(i) = MIgrad(x, theta, xtrain_norm, ctrain, xnorm(:,i),model, mu_c(i), bald(i));
end

idx= find(L==max(L));
if numel(idx)~=1
    idx = randsample(idx,1);
end

new_x = x(:,idx);

new_x_norm = (new_x - model.lb)./(model.ub - model.lb);
idx = find(ismember(xog',new_x', 'rows'));

end
function MIgrad_x = MIgrad(x, theta, xtrain_norm, ctrain, xnorm,model, mu_c, L)
c0 = [ctrain, 0];
c1 = [ctrain, 1];
post = [];
L0 = BALD(theta, [xtrain_norm, xnorm], c0, xnorm, model, post);
L1 = BALD(theta, [xtrain_norm, xnorm], c1, xnorm, model, post);
MIgrad_x = L-(mu_c.*L1 + (1-mu_c).*L0);
end

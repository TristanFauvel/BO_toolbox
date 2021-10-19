function [new_x,new_x_norm, idx, L] = BALD_grid(x, theta, xtrain_norm, ctrain,model, post)
xog = x;
ngrid = 300;
if size(x,2)>ngrid
    keep = randsample(size(x,2), ngrid);
    x = x(:,keep);
end

xnorm = (x - model.lb)./(model.ub - model.lb);

L = BALD(theta, xtrain_norm, ctrain, xnorm, model, post);

idx= find(L==max(L));
if numel(idx)~=1
    idx = randsample(idx,1);
end

new_x = x(:,idx);

new_x_norm = (new_x - model.lb)./(model.ub - model.lb);
idx = find(ismember(xog',new_x', 'rows'));

return
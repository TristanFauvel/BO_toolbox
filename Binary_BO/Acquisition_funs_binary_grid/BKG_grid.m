function [new_x, new_x_norm] = BKG_grid(theta, xtrain_norm, ctrain,model, post, approximation)

nx = 30;
d = size(xtrain_norm,1);

if d== 1
    xtest= linspace(lb_norm(1),ub_norm(1),nx);
elseif d == 2
    x_range_1 = linspace(lb_norm(1),ub_norm(1),nx);
    x_range_2 = linspace(lb_norm(2),ub_norm(2),nx);
    [p,q] = ndgrid(x_range_1,x_range_2);
    xtest = [p(:), q(:)]';
else
    error('Binary KG should not be used with high dimensioinal inputs')
end
N = size(xtest,2);
c0 = [ctrain, 0];
c1 = [ctrain,1];

regularization = 'nugget';
[mu_c,  mu_y] =  model.prediction(theta, xtrain_norm, ctrain, xtest, post);
gmax = max(mu_y);

N= size(mu_c,1);
U = zeros(1,N);
for i = 1:N
    [mu_c0,  mu_y0] =  model.prediction(theta, [xtrain_norm, xtest(:,i)], c0, xtest, post);
    [mu_c1,  mu_y1] =  model.prediction(theta, [xtrain_norm, xtest(:,i)], c1, xtest, post);
    u = (max(mu_y0)-gmax).*(1-mu_c(i)) + (max(mu_y1)-gmax).*mu_c(i);
    U(i) = u;
end
[a,idx]= max(U);
if numel(idx)~=1
    idx = randsample(idx,1);
end
new_x_norm = xtest(:,idx);
new_x = new_x_norm.*(model.max_x-model.min_x) + model.min_x;

return

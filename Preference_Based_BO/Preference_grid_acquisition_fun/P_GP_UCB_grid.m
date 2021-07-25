function new_duel = P_GP_UCB_grid(x, theta, xtrain, ctrain, kernelfun,modeltype, m,  kernelname)

[d,n] = size(x);

[p,q]= meshgrid(x);
xduels = [p(:), q(:)]';

[~, value, ~, ~] = prediction_bin(theta, xtrain, ctrain, [x; 0.5*ones(d,n)], kernelfun, kernelname,modeltype, post, regularization);
sum_value = value + value';
[mu_c,  mu_y, ~, Sigma2_y] = prediction_bin(theta, xtrain, ctrain, xduels, kernelfun, kernelname,modeltype, post, regularization);

samples =mvnrnd(mu_y, Sigma2_y, 1000);%sample form the p(f|D)
% V = mean(link(samples).^2,1)' - link(mu_y./sqrt(1+sigma2_y)); % In case were link = sigmoid, replace with : link(my_s./sqrt(1+sy_2_s*pi/8)).^2;
V = mean(normcdf(samples).^2,1)' - mu_c.^2; 

epsilon= 1;
alpha =  sum_value(:) + epsilon*V;
maxid= find(alpha==max(alpha(:)));
if numel(maxid)~=1
    maxid = randsample(maxid,1);    
end
[i,j] = ind2sub([n,n],maxid);
new_duel = [x(:,i), x(:,j)];



% figure()
% imagesc(reshape(alpha, n, n))
% colorbar
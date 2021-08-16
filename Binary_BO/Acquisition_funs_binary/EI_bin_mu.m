function new_x = EI_bin_mu(theta, xtrain_norm, ctrain, kernelfun, kernelname, modeltype,max_x, min_x, lb_norm, ub_norm)
% Inspired by the expected improvement criterion by Tesch et al 2013

nx = 40;
ns = 40;
x_range = linspace(0,1,nx);
s_range = linspace(0,1,ns);

[p,q]= ndgrid(s_range, x_range);

xtest = [p(:),q(:)]';

[mu_c,  mu_y, sigma2_y,Sigma2_y] =  prediction_bin(theta, xtrain_norm, ctrain, xtest, model, post);
mu_c = reshape(mu_c, ns,nx);
mu_y = reshape(mu_y, ns,nx);
sigma2_y = reshape(sigma2_y, ns,nx);
[mu_c_max,b] = max(mu_c, [], 2);

% figure(); plot(mu_c_max);
N = 10000;
U = NaN(ns,nx);
for i=1:nx
    for j = 1:model.ns
        %samps =mu_y(j,i) + sqrt(sigma2_y(j,i))*randn(1, N);
        %U(j,i) = mean(max(normcdf(samps)-mu_c(j,i),0));
        mu = [mu_y(j,i), mu_c_max(j)];
        k = sub2ind([ns,nx],j,i);
        ind = [k,sub2ind([ns,nx],j,b(j))];
        S = Sigma2_y(ind,ind);
        samps = mvnrnd(mu, S, N);
        U(j,i) = mean(max(normcdf(samps(:,1))- normcdf(samps(:,2)),0));
    end
end

U = U(:);
[a,idx]= max(U);
if numel(idx)~=1
    idx = randsample(idx,1);
end
new_x = xtest(:,idx);
return


figure()
imagesc(s_range, x_range, reshape(mu_c, ns,nx)); hold on;
set(gca,'YDir','normal')
xlabel('x')
ylabel('s')
pbaspect([1,1,1])
title('$\mu_c$')
colorbar
scatter(xtrain_norm(2, ctrain == 0), xtrain_norm(1, ctrain == 0),'k','filled'); hold on;
scatter(xtrain_norm(2, ctrain == 1), xtrain_norm(1, ctrain == 1),'r', 'filled'); hold off;


figure()
subplot(1,3,1)
imagesc(s_range, x_range, reshape(mu_y, ns,nx)); hold on;
set(gca,'YDir','normal')
xlabel('x')
ylabel('s')
pbaspect([1,1,1])
title('$\mu_y$')
colorbar
scatter(xtrain_norm(2, ctrain == 0), xtrain_norm(1, ctrain == 0),'k','filled'); hold on;
scatter(xtrain_norm(2, ctrain == 1), xtrain_norm(1, ctrain == 1),'r', 'filled'); hold off;
subplot(1,3,2)
imagesc(x_range, s_range, reshape(sigma2_y, ns,nx)); hold on;
set(gca,'YDir','normal')
pbaspect([1,1,1])
xlabel('x')
ylabel('s')
title('$\sigma^2_y$')
colorbar
scatter(xtrain_norm(2, ctrain == 0), xtrain_norm(1, ctrain == 0),'k','filled'); hold on;
scatter(xtrain_norm(2, ctrain == 1), xtrain_norm(1, ctrain == 1),'r', 'filled'); hold off;
subplot(1,3,3)
imagesc(s_range, x_range, reshape(U, ns,nx)); hold on;
set(gca,'YDir','normal')
pbaspect([1,1,1])
xlabel('x')
ylabel('s')
title('EI')
colorbar
scatter(xtrain_norm(2, ctrain == 0), xtrain_norm(1, ctrain == 0),'k','filled'); hold on;
scatter(xtrain_norm(2, ctrain == 1), xtrain_norm(1, ctrain == 1),'r', 'filled'); hold off;


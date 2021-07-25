function [I, dIdx]= BALD(theta, xtrain, ctrain, kernelfun, modeltype, post)

regularization = 'nugget';
if nargout == 1
    [mu_c,  mu_y, sigma2_y] =  prediction_bin(theta, xtrain, ctrain, x, kernelfun, modeltype, post, regularization);
else
    [mu_c,  mu_y, sigma2_y, ~, ~, dmuy_dx, dsigma2y_dx] =  prediction_bin(theta, xtrain, ctrain, x, kernelfun, modeltype, post, regularization);
end
h = @(p) -p.*log(p+eps) - (1-p).*log(1-p+eps);

% for a gaussian cdf link function:
C = sqrt(pi*log(2)/2);

I1 = h(mu_c);
I2 =  log(2)*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
I = I1 - I2;
I = -I; % For minimization

%for a sigmoid link
%C = sqrt(2*log(2));
%I = h(mu_c) - 2*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
if nargout>1
dhdp = @(p) -log(p)+log(1-p);
arg = mu_y./sqrt(sigma2_y+C^2);
dI1dx = (((dmuy_dx).*sqrt(sigma2_y+C^2)-mu_y.*dsigma2y_dx./(2*sqrt(sigma2_y+C^2)))./(sigma2_y+1)).*normpdf(arg).*dhdp(normcdf(arg));

dI2dx =I2.*(0.5*mu_y.^2.*dsigma2y_dx-mu_y.*(sigma2_y+C^2).*dmuy_dx)./((sigma2_y+C^2).^2)-I2./(2*(sigma2_y+C^2)).*dsigma2y_dx;
% dI2dx = dI2dx(1:d,:);

dIdx = dI1dx - dI2dx;
dIdx = -dIdx;
dIdx = squeeze(dIdx);
end

return

n = 100;
mu_y_range = linspace(-10,10,n);
sigma2_y_range= linspace(0,100,n);

[p,q] = meshgrid(mu_y_range, sigma2_y_range);
mu_y = p(:);
sigma2_y = q(:);
mu_c = normcdf(mu_y./sqrt(sigma2_y+1));
C = sqrt(pi*log(2)/2);

I1 = h(mu_c);
I2 =  log(2)*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
I = I1 - I2;

figure()
subplot(1,2,1)
imagesc(mu_y_range, sigma2_y_range, reshape(mu_c,n,n));
set(gca,'YDir','normal')
xlabel('$\mu_y$')
ylabel('$\sigma^2_y$')
title('$\mu_c$')
pbaspect([1,1,1])
subplot(1,2,2)
imagesc(mu_y_range, sigma2_y_range, reshape(I,n,n));
set(gca,'YDir','normal')
xlabel('$\mu_y$')
ylabel('$\sigma^2_y$')
pbaspect([1,1,1])

figure()
scatter(mu_c, I)

figure()
scatter(I,mu_c)

%%
sigma2_y_range=0; % linspace(0,100,n);

[p,q] = meshgrid(mu_y_range, sigma2_y_range);
mu_y = p(:);
sigma2_y = q(:);
mu_c = normcdf(mu_y./sqrt(sigma2_y+1));
C = sqrt(pi*log(2)/2);

I1 = h(mu_c);
I2 =  log(2)*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
I = I1 - I2;

figure()
plot(normcdf(mu_y_range),I)
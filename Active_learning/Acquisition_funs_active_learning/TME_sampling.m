function  [new_x, idx, L] = TME_sampling(x, theta, xtrain, ctrain,kernelfun, modeltype)

%% Direct method
nx = size(x,2);
xrange = x ; %linspace(0,1,nx);
gaussian_entropy = @(Sigma) 0.5+log(det(Sigma)) + 0.5*size(Sigma,1)*log(2*pi*exp(1));

[mu_c,  mu_y, sigma2_y] =  prediction_bin(theta, xtrain, ctrain, xrange, kernelfun, 'modeltype', modeltype);

%% Total marginal entropy
H1 = 0;
for i = 1:nx
    H1 = H1 + gaussian_entropy(sigma2_y(i));
end

% [mu_c,  mu_y, sigma2_y] =  prediction_bin(theta, xtrain, ctrain, x, kernelfun, 'modeltype', modeltype, 'post', post);

%% Expected total marginal entropy after observing y
c0 = [ctrain,0];
c1 = [ctrain,1];
H2 = zeros(numel(x),1);
for i =1:numel(x)
    %case y = 0
    [~, ~, sigma2_y0] =  prediction_bin(theta, [xtrain, x(:,i)], c0, xrange, kernelfun, 'modeltype', modeltype);
    %case y = 1
    [~,~, sigma2_y1] =  prediction_bin(theta, [xtrain, x(:,i)], c1, xrange, kernelfun, 'modeltype', modeltype);
    H20 = 0;
    H21 = 0;
    for j = 1:nx
        H20= H20 +  gaussian_entropy(sigma2_y0(j));
        H21= H21 + gaussian_entropy(sigma2_y1(j));
    end
    H2(i) = mu_c(i)*H21 + (1-mu_c(i))*H20;
end
I= H1-H2;

L = I;
[a,idx] = max(L);
new_x = x(:,idx);
dIdx =[];

return

figure()
plot(xrange, I/max(I)); hold on
plot(xrange, mu_c); hold off;

figure()
plot(sigma2_y); hold on;
plot(sigma2_y0);  hold on;
plot(sigma2_y1); hold on;

%% Direct computation of BALD :
x = linspace(0,1,100);
xrange = x;
[mu_c,  mu_y, sigma2_y, Sigma2_y] =  prediction_bin(theta, xtrain, ctrain, x, kernelfun, 'modeltype', modeltype);
H1 = gaussian_entropy(Sigma2_y);
%% Expected total marginal entropy after observing y
c0 = [ctrain,0];
c1 = [ctrain,1];
H2 = zeros(numel(x),1);
for i =1:numel(x)
    %case y = 0
     [~,~,~, Sigma2_y0]  =  prediction_bin(theta, [xtrain, x(:,i)], c0, xrange, kernelfun, 'modeltype', modeltype);
    %case y = 1
    [~,~,~, Sigma2_y1]  =  prediction_bin(theta, [xtrain, x(:,i)], c1, xrange, kernelfun, 'modeltype', modeltype);
    H20 = gaussian_entropy(Sigma2_y0);
    H21 = gaussian_entropy(Sigma2_y1);
    
    H2(i) = mu_c(i)*H21 + (1-mu_c(i))*H20;
end
I= H1-H2;
figure()
plot(real(I))
%%
% h = @(p) -p.*log(p+eps) - (1-p).*log(1-p+eps);
% % for a gaussian cdf link function:
% C = sqrt(pi*log(2)/2);
%
% nx = 30;
% x_range = linspace(0,1,nx);
%
% [mu_c,  mu_y, sigma2_y] =  prediction_bin(theta, xtrain, ctrain, xrange, kernelfun, 'modeltype', modeltype, 'post', post);
% I1 = sum(h(mu_c));
% I2 = 0;
% for x = xrange
%     ckfun = conditioned_kernelfun(theta, kernelfun, xi, xj, training, x0, reg)
%
%     I2 = I2 + log(2)*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
% end
%
%
%
%
% I1 = h(mu_c);
% I2 =  log(2)*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
% I = I1 - I2;
% I = -I; % For minimization
%
% %for a sigmoid link
% %C = sqrt(2*log(2));
% %I = h(mu_c) - 2*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
% if nargout>1
% dhdp = @(p) -log(p)+log(1-p);
% arg = mu_y./sqrt(sigma2_y+C^2);
% dI1dx = (((dmuy_dx).*sqrt(sigma2_y+C^2)-mu_y.*dsigma2y_dx./(2*sqrt(sigma2_y+C^2)))./(sigma2_y+1)).*normpdf(arg).*dhdp(normcdf(arg));
%
% dI2dx =I2.*(0.5*mu_y.^2.*dsigma2y_dx-mu_y.*(sigma2_y+C^2).*dmuy_dx)./((sigma2_y+C^2).^2)-I2./(2*(sigma2_y+C^2)).*dsigma2y_dx;
% % dI2dx = dI2dx(1:d,:);
%
% dIdx = dI1dx - dI2dx;
% dIdx = -dIdx;
% dIdx = squeeze(dIdx);
% end
%


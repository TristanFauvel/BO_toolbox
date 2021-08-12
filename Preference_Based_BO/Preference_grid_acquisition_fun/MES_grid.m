function new_duel = MES_grid(x, theta, xtrain, ctrain, kernelfun,modeltype, m, kernelname)

%(xrange, theta, xtrain, ctrain, kernelfun, link, xduels,  mu_y_acq, sigma2_y_acq, Sigma2_y_acq, modeltype, C, mu_c_acq)

x0 = xrange(1);
% Compute the posterior over f for the training points
x= xrange;
xp=xrange;
n= size(x,2);
np= size(xp,2);
[mu_c,  mu_y, sigma2_y, Sigma2_y] = prediction_bin(theta, xtrain, ctrain, xtrain, kernelfun, kernelname modeltype, post, regularization);
[~,  g_mu_y, g_sigma2_y, g_Sigma2_y] = prediction_bin(theta, xtrain, ctrain, [x;x0*ones(1,n)], kernelfun, kernelname, modeltype, post, regularization);

h = @(p) -p.*log(p+eps) - (1-p).*log(1-p+eps); %entropy of a binary variable

nsamples=60;
gmax=zeros(1, nsamples);
g_sigma_y= sqrt(g_sigma2_y);

ent = zeros(n, np);
for k = 1:model.nsamples
    %sample the maximum
    fsamples = mvnrnd(mu_y,  nearestSPD(Sigma2_y))'; %sample from the posterior at training points
    sample_g = sample_value_GP(xrange, theta, xtrain, fsamples, Sigma2_y);
    gmax(k)= max(sample_g); %sample g* from p(g*|D)
end
save('mu_range.mat', 'g_mu_y')
save('sigma2_range.mat', 'g_Sigma2_y')
save('ub.mat', 'gmax')

[status,cmdout] =system('Rscript batch_compute_moments.R ');

ent= load('ent.mat');
ent=ent.ent;
s = ent+ent';
s= s+h(0.5)*eye(n);

delete 'ent.mat'

mu_c_acq_r = reshape(mu_c_acq, n,n);

acq= h(mu_c_acq_r) - s; %Note : it should be symmetric
%acq(boolean(eye(n))) = 0;

[a,b]= max(acq(:));
new_duel = xduels(:,b);

% figure()
% imagesc(mu_c_acq_r);
% colorbar
%
% figure()
% imagesc( h(mu_c_acq_r));
% colorbar
%
%
% figure()
% imagesc(s);
% colorbar
%
% figure()
% imagesc(acq);
% colorbar
%

function new_duel= Copeland_duel_thompson_grid(x, theta, xtrain, ctrain, kernelfun,modeltype, m, kernelname, post)

if m > 2 
error('Duel Thompson sampling only possible with duels, not tournaments')
end
%(x, theta, xtrain, ctrain, kernelfun, link, xduels,  mu_y_acq, sigma2_y_acq, Sigma2_y_acq, modeltype, maxC, mu_c_acq);

%sample a preference function f according to the predicted mean and covaraince
[mu_c,  mu_y, sigma2_y, Sigma2_y] = prediction_bin(theta, xtrain, ctrain, xtrain, kernelfun,kernelname, modeltype, post, regularization);

fsamples = mvnrnd(mu_y,  nearestSPD(Sigma2_y))'; %sample from the posterior at training points
Sigma2 = exp(-15); % To regularize

xtrain_extended = [xtrain,flipud(xtrain)]; %To enforce symmetry;
fsamples_extended= [fsamples;-fsamples];
nFeatures =500;
f_tilde = sample_GP(xduels, theta, xtrain_extended, fsamples_extended, Sigma2,  kernelname, 'post', post);%sample_GP_approximation_with_noise_for_gaussian_kernel(xtrain_extended', fsamples_extended, theta, nFeatures); %This is not precisely described in the paper... but here I take as values of y in the (x, y) pairs a sample from the posterior.
sample_pi_f_tilde = link(f_tilde); %@(x) link(f_tilde(x));

[d, ntest]= size(x);


C= soft_copeland_score(reshape(sample_pi_f_tilde, ntest, ntest));

idmaxC= find(C==max(C));
if numel(idmaxC)~=1
    idmaxC = randsample(idmaxC,1);    
end

new_duel = NaN(d*2,1);
new_duel(1:d,1) = x(:,idmaxC);


%Monte-Carlo estimation of the variance of mu_c
ax= ismember(xduels(1:d,:)', new_duel(1:d)', 'rows')';
my_s = mu_y_acq(ax);
sy_2_s = sigma2_y_acq(ax);
Sy_s = Sigma2_y_acq(ax,ax);
mu_c_s=  mu_c_acq(ax);

samples =mvnrnd(my_s, nearestSPD(Sy_s), 100000);%sample form the p(f|D)

V = mean(link(samples).^2,1)' - mu_c_s.^2; 

maxid= find(V==max(V));
if numel(maxid)~=1
    maxid = randsample(maxid,1);    
end

new_duel(d+1:end,1) = x(:,maxid);

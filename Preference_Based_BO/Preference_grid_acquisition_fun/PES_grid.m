function new_phi = new_DTS_grid(x, theta, xtrain, ctrain, kernelfun,modeltype, m, kernelname)

%(x, theta, xtrain, ctrain, kernelfun, link, x2d_test, modeltype)
h = @(p) -p.*log(p+eps) - (1-p).*log(1-p+eps); %entropy of a binary variable

%sigma0, sigma, l are the hyperparameters
% We sample from the global maximum
nFeatures = 1000;
%ytrain should be a column vector
%xtrain should be of size n x nd


fsamples = mvnrnd(mu_y,  Sigma2_y)'; %sample from the posterior at training points
xtrain_extended = [xtrain,flipud(xtrain)]; %To enforce symmetry;
fsamples_extended= [fsamples;-fsamples];
[f_tilde df_tilde] = sample_GP_approximation_with_noise_for_gaussian_kernel(xtrain_extended', fsamples_extended, theta, nFeatures); %This is not precisely described in the paper... but here I take as values of y in the (x, y) pairs a sample from the posterior.
sample_pi_f_tilde = @(x) link(f_tilde(x));

Cfun = @(x) mean(sample_pi_f_tilde([x*ones(1,nplot);xp]')); %compute the sampled copeland function
dCdx = @(x) mean(dlink(sample_pi_f_tilde([x*ones(1,nplot);xp]')).*df_tilde([x*ones(1,nplot);xp]')); %does not work yet


% We call the ep method

ret = initializeEPapproximation(Xsamples, Ysamples, m, l, sigma, sigma0, hessians);

% We define the cost function to be optimized

cost = @(x) evaluateEPobjective(ret, x);

% We optimize globally the acquisition function

optimum = globalOptimizationOneArgument(cost, xmin, xmax, guesses);

PES = @(x) h(mu_c(x)) ;

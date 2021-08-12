function new_duel = MVES_grid(x, theta, xtrain, ctrain, kernelfun,modeltype, m, kernelname)
%(xrange, theta, xtrain, ctrain, kernelfun, link, xduels,  mu_y_acq, sigma2_y_acq, Sigma2_y_acq, modeltype, C, mu_c_acq)
% Monte Carlo method

x0 = xrange(1);
% Compute the posterior over f for the training points
x= xrange;
xp=xrange;
n= size(x,2);
np= size(xp,2);
[mu_c,  mu_y, sigma2_y, Sigma2_y] = prediction_bin(theta, xtrain, ctrain, xtrain, kernelfun, kernelname,modeltype, post, regularization);
[~,  g_mu_y, g_sigma2_y, g_Sigma2_y] = prediction_bin(theta, xtrain, ctrain, [x;x0*ones(1,n)], kernelfun, kernelname, modeltype, post, regularization);


h = @(p) -p.*log(p+eps) - (1-p).*log(1-p+eps); %entropy of a binary variable

nsamples=60;


nint= 30;

int=0;
for k = 1:model.nsamples
    
    %sample the maximum
    fsamples = mvnrnd(mu_y,  nearestSPD(Sigma2_y))'; %sample from the posterior at training points
    sample_g = sample_value_GP(xrange, theta, xtrain, fsamples, Sigma2_y);
    max_g= max(sample_g); %sample g* from p(g*|D)
       
    %% Compute p(y=1|x,x',g*)
    P_y = zeros(n,np);
    
    
    for ix = 1:n
        for ixp = ix+1:np %1:np
            %compute the mean and variance of p(g(x'),g(x)|D, g*)
            mu = g_mu_y([ix,ixp]);
            Sigma = [g_Sigma2_y(ix,ix) , g_Sigma2_y(ix,ixp);g_Sigma2_y(ixp,ix), g_Sigma2_y(ixp,ixp)];
            %Sample from p(g(x), g(x')|D, g*)
            X=mvrandn([-inf; -inf], [max_g; max_g]-mu, Sigma,nint)+mu;
            P_y(ix,ixp) = mean(link(X(1,:) - X(2,:)));
        end
    end
    int = int + h(P_y);%second term (expectation of the entropy of y|D,x,xp,g*
end
s = int/nsamples;



mu_c_acq_r = reshape(mu_c_acq, n,n);
% mu_c_acq_r = mu_c_acq_r(2:end,2:end);

acq= h(mu_c_acq_r) - (s+s'); %Note : it should be symmetric
acq(boolean(eye(n))) = 0;

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

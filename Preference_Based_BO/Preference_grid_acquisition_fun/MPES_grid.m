function new_duel = MPES_grid(x, theta, xtrain, ctrain, kernelfun,modeltype, m, kernelname)

%(x, theta, xtrain, ctrain, kernelfun, link, xduels,  mu_y_acq, sigma2_y_acq, Sigma2_y_acq, modeltype, C, mu_c_acq)
%Max-preference entropy search
[mu_c,  mu_y, sigma2_y, Sigma2_y] = prediction_bin(theta, xtrain, ctrain, xtrain, kernelfun, modeltype, post, regularization);
h = @(p) -p.*log(p+eps) - (1-p).*log(1-p+eps); %entropy of a binary variable

nsamples=100;
ystar= zeros(1,nsamples);
samples_g=zeros(nsamples,numel(x));

ent= zeros(size(xduels,2), 1);
py = zeros(size(xduels,2), nsamples);

for k = 1:nsamples
    fsamples = mvnrnd(mu_y,  nearestSPD(Sigma2_y))'; %sample from the posterior at training points
    %sample_g= sample_value_GP(x, theta, xtrain, fsamples, Sigma2_y);
    %samples_g(k,:)=sample_g;
    sample_f = sample_preference_GP(x, theta, xtrain, fsamples, Sigma2_y);

    %[maxg, idxmaxg]= max(sample_g);
    [maxf, idxmaxf]= max(sample_f(:));

    approx = 'yes';
    if strcmp(approx, 'yes')
        % Laplace approximation (moment matching)
        %                         ratio = normpdf(maxf)./normcdf(maxf);
        %                         mu_tilde= mu_y_acq- sqrt(sigma2_y_acq)*ratio;
        %                         sigma2_tilde = sigma2_y_acq.*(1- maxf*ratio-ratio.^2);
        %                         py(:,k) = normcdf(mu_tilde./sqrt(sigma2_tilde+1))-normcdf(maxf)*(1-normcdf((maxf-mu_tilde)./sqrt(sigma2_tilde)));
        %
        % Approximation
        py(:,k) = normcdf(mu_y_acq./sqrt(sigma2_y_acq+1))-normcdf(maxf)*(1-normcdf((maxf-mu_y_acq)./sqrt(sigma2_y_acq)));
        py(:,k) = py(:,k)./(normcdf((maxf-mu_y_acq)./sqrt(sigma2_y_acq)));
        py(py(:,k)>1,k)=1;% this is cheating
    else
        %% sample from truncated gaussian
        for m = 1:100
            X=trandn(-inf*ones(numel(mu_y_acq),1),(maxf*ones(numel(mu_y_acq),1)-mu_y_acq)./sqrt(sigma2_y_acq)); %sample from a truncated gaussian and set Z=m+s*X
            Z=mu_y_acq+sqrt(sigma2_y_acq).*X; %sample Z from a gaussian (m, s^2) conditional on -inf<Z<u
            py(:,k) = py(:,k)+ normcdf(Z);
        end
        py(:,k) = py(:,k)/m;
    end
    ent(:) = ent + h(py(:,k));
end

MES= h(mu_c_acq) - ent/nsamples; %Note : it should be symmetric
[a,b]= max(MES);
new_duel = xduels(:,b);

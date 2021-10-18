function new_x = batch_MUC(theta, xtrain_norm, ctrain, model, post, ~)
options.method = 'lbfgs';
options.verbose = 1;
D = size(xtrain_norm,1)/2;
ncandidates =5;
init_guess = [];

% x_best_norm = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm, model.ub_norm, ncandidates,init_guess, options);
x_best_norm = post.x_best_norm;

new = multistart_minConf(@(x)pref_MI(theta, xtrain_norm, ctrain, x_best_norm , x, model, post), repmat(model.lb_norm,model.nsamples-1,1), repmat(model.ub_norm,model.nsamples-1,1), ncandidates,init_guess, options);

xnorm = [x_best_norm; new];
xnorm = reshape(xnorm, D, model.nsamples);
new_x = xnorm.*(model.ub-model.lb) + model.lb;
end

function [mu_c, dmuc_dx] = pref_MI(theta, xtrain_norm, ctrain, x_best_norm, xin, model, post) %[sigma2_y, dsigma2y_dx]
D = size(x_best_norm,1);
nd = 2*D;
x = [x_best_norm, reshape(xin,D, model.nsamples-1)];
m = model.nsamples;
modeltype = model.modeltype;
% Compute all possible outcomes

if strcmp(model.feedback, 'all')
    % Compute equivalent duels
    iduels = nchoosek(1:model.nsamples,2)';
    nduels = size(iduels,2);
    
    xduels = reshape(x(:,iduels(:)), 2*model.D, nduels);
    
    V = [];
    for i = 1:nduels
        u = [ones(1,i-1), zeros(1, nduels-i+1)] ;
        V = [V;unique(perms(u),'rows')];
    end
    
    %     if nargout>1
    [mu_c,  mu_y, sigma2_y, Sigma2_y, dmucdx, dmuydx, dsigma2ydx, dSigma2y_dx] =  model.prediction(theta, xtrain_norm, ctrain, xduels, post);
    
    
    dSigma2y = zeros(nduels, nduels, nd);
    dmuc_dx = zeros(nd,nduels);
    dmuy_dx = zeros(nd,nduels);
    dsigma2y_dx = zeros(nd,nduels);
    
    for i = 1:nduels
        vals = squeeze(dSigma2y_dx(:,i,i,:));
        vals(iduels' == 1)=0;
        dSigma2y(:,i,:) = vals;
        dmuc_dx(:,i) = dmucdx(i,i,:);
        dmuy_dx(:,i) = dmuydx(i,i,:);
        dsigma2y_dx(:,i) = dsigma2ydx(i,i,:);
    end
    dmuc_dx = dmuc_dx';
    dmuy_dx = dmuy_dx';
    dsigma2y_dx = dsigma2y_dx';
    
    dmuc_dx(iduels' == 1)=0;
    dmuy_dx(iduels' == 1)=0;
    dsigma2y_dx(iduels' == 1)=0;
    %     else
    %         [mu_c,  mu_y, sigma2_y, Sigma2_y] =  model.prediction(theta, xtrain_norm, ctrain, xduels, post);
    %     end
end


if strcmp(modeltype, 'exp_prop')
    % for a gaussian cdf link function:
    C = sqrt(pi*log(2)/2);
    I2 =  log(2)*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
elseif strcmp(modeltype, 'laplace')
    %for a sigmoid link
    C = sqrt(2*log(2));
    I2 = 2*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
end
%I1 = h(mu_c);


nsamples = 1e4;
samples= mvnrnd(mu_y,Sigma2_y, nsamples);
phi_samps = normcdf(samples);

est = zeros(size(V,1),1);


for i = 1:size(V,1)
    arg  = prod(phi_samps.*V(i,:) + (1- phi_samps).*(1-V(i,:)),2);
    est(i) = mean(arg);
    
end
I1 = -sum(est.*log(est));

I = I1 - sum(I2);

% if nargout >1


[L, dLdx] = log_mvngauss(samples', mu_y, Sigma2_y, dmuy_dx, dSigma2y);

destdx = zeros(size(V,1), nduels);
for i = 1:size(V,1)
    arg  = prod(phi_samps.*V(i,:) + (1- phi_samps).*(1-V(i,:)),2);
    destdx(i,:) = mean(arg.*dLdx,1);
    
end

dI1dx = -sum((1+est).*destdx,1);

dI2dx =I2.*(0.5*mu_y.^2.*dsigma2y_dx-mu_y.*(sigma2_y+C^2).*dmuy_dx)./((sigma2_y+C^2).^2)-I2./(2*(sigma2_y+C^2)).*dsigma2y_dx;
dI2dx = sum(dI2dx, 1);

dIdx = dI1dx - dI2dx;
dIdx= dIdx(:);
% end
I2 = sum(I2);

end
%
% f = @(x) pref_MI(theta, xtrain_norm, ctrain, x_best_norm, x, model, post)
% N = 100;
% x = rand((model.nsamples-1)*D,N);
% num_res = NaN((model.nsamples-1)*D,N);
% res = NaN((model.nsamples-1)*D,N);
% for i = 1:N
%     num_res(:,i) = squeeze(test_matrix_deriv(f,x(:,i),1e-9));
%     [~, res(:,i)] = pref_MI(theta, xtrain_norm, ctrain, x_best_norm, x(:,i), post);
% end
% figure();
% plot(num_res(:)); hold on ;
% plot(res(:)); hold off;
%
% %
% % %%
% f = @(x) pref_MI(theta, xtrain_norm, ctrain, x_best_norm, x, model, post)
% N = 100;
% x = rand((model.nsamples-1)*D,N);
% num_res = NaN(3,3,4,N);
% res = NaN(3,3,4,N);
% for i = 1:N
%     num_res(:,:,:,i) = squeeze(test_matrix_deriv(f,x(:,i),1e-9));
%     [~, res(:,:,:,i)] = pref_MI(theta, xtrain_norm, ctrain, x_best_norm, x(:,i), post);
% end
% figure();
% plot(num_res(:)); hold on ;
% plot(res(:)); hold off;
%
% %
% f = @(x) pref_MI(theta, xtrain_norm, ctrain, x_best_norm, x, model, post)
% N = 100;
% x = rand((model.nsamples-1)*D,N);
% num_res = NaN(3,2*D,N);
% res = NaN(3,2*D,N);
% for i = 1:N
%     num_res(:,:,i) = squeeze(test_matrix_deriv(f,x(:,i),1e-9));
%     [~, res(:,:,i)] = pref_MI(theta, xtrain_norm, ctrain, x_best_norm, x(:,i), post);
% end
% figure();
% plot(num_res(:)); hold on ;
% plot(res(:)); hold off;
%

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

function [L, dLdx] = pref_MI(theta, xtrain_norm, ctrain, x_best_norm, xin, model, post) %[sigma2_y, dsigma2y_dx]
D = size(x_best_norm,1);
nd = 2*D;
x = [x_best_norm, reshape(xin,D, model.nsamples-1)];
m = model.nsamples;
modeltype = model.modeltype;
% Compute all possible outcomes

if ~strcmp(model.feedback, 'all')
    error('Function only implemented for the all version')
end

% Compute equivalent duels
iduels = nchoosek(1:model.nsamples,2)';
nduels = size(iduels,2);

xduels = reshape(x(:,iduels(:)), 2*model.D, nduels);

V = [];
for i = 1:nduels
    u = [ones(1,i-1), zeros(1, nduels-i+1)] ;
    V = [V;unique(perms(u),'rows')];
end

%% Samples from GP on the value function
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmucdx, dmuydx, dsigma2ydx, dSigma2y_dx] =  model.prediction(theta, xtrain_norm, ctrain,...
    [x; model.condition.x0*ones(size(x))],model, post);


dSigma2y = zeros(model.nsamples, model.nsamples, D);
dmuc_dx = zeros(D,model.nsamples-1);
dmuy_dx = zeros(D,model.nsamples-1);
dsigma2y_dx = zeros(D,model.nsamples-1);

for i = 1:model.nsamples
    vals = squeeze(dSigma2y_dx(:,i,i,1:D));
    vals(iduels' == 1)=0;
    dSigma2y(:,i,:) = vals;
    dmuc_dx(:,i) = dmucdx(i,i,1:D);
    dmuy_dx(:,i) = dmuydx(i,i,1:D);
    dsigma2y_dx(:,i) = dsigma2ydx(i,i,1:D);
end
dmuc_dx = dmuc_dx';
dmuy_dx = dmuy_dx';
dsigma2y_dx = dsigma2y_dx';

% dmuc_dx(iduels' == 1)=0;
% dmuy_dx(iduels' == 1)=0;
% dsigma2y_dx(iduels' == 1)=0;
% 
nsamples = 10e4;
 samples= mvnrnd(mu_y,Sigma2_y, nsamples);
% samples= mu_y'+0.1;
%% Compute the derivatives of dlog(p(f|D))/dx
[L, dLdx] = log_mvngauss(samples', mu_y, Sigma2_y, dmuydx(:,:,1:D), dSigma2y_dx(:,:,:,1:D));

%% Compute I1
sampled_duels_1 =samples(:,iduels(1,:));
sampled_duels_2 =samples(:,iduels(2,:));

phi_samps = normcdf(sampled_duels_1-sampled_duels_2);
est = zeros(size(V,1),1);
destdx = zeros(size(V,1), model.nsamples);
for i = 1:size(V,1)
    arg  = prod(phi_samps.*V(i,:) + (1- phi_samps).*(1-V(i,:)),2);
    est(i) = mean(arg);    
        destdx(i,:) = mean(arg.*dLdx,1);
end
logest = log(est);
I1 = -sum(est.*logest);
dI1dx = -sum(1+logest.*destdx,1)';


%% Compute I2 and its derivatives
[mu_c,  mu_y, sigma2_y] =  model.prediction(theta, xtrain_norm, ctrain, xduels, post);

if strcmp(modeltype, 'exp_prop')
    % for a gaussian cdf link function:
    C = sqrt(pi*log(2)/2);
    I2_duels =  log(2)*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
elseif strcmp(modeltype, 'laplace')
    %for a sigmoid link
    C = sqrt(2*log(2));
    I2_duels = 2*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
end
I2 = sum(I2_duels);

dI2dx = zeros(model.nsamples, D);

for i = 1:(model.nsamples-1)
    d = iduels(:,i);
    dI2dx(d,:) = dI2dx(d) + I2_duels(d).*(0.5*mu_y(d).^2.*dsigma2y_dx(d,:)-mu_y(d).*(sigma2_y(d)+C^2).*dmuy_dx(d,:))./((sigma2_y(d)+C^2).^2)-I2_duels(d)./(2*(sigma2_y(d)+C^2)).*dsigma2y_dx(d,:);
end
    
% dI2dx = sum(dI2dx(2:end,:), 2);
% dI2dx = dI2dx(2:end,:);


%% Compute I and its derivatives
  
I = I1 - I2;
 
dI2dx = dI2dx(2:end,:);
 dI1dx =  dI1dx(2:end,:);
 
dIdx = dI1dx - dI2dx;

 dLdx =  dLdx(1:3);
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

D = 1;
N= 20;
x = rand(D,N);
regularization = 'nugget';
dmuc_dx = NaN(D,N);
dmuy_dx = NaN(D,N);
dsigma2y_dx = NaN(D,N);

drdx = NaN(D,N);
dSigma2y_dx_num = NaN(D,N,N);

xtrain_norm = rand(D,2*N);
ctrain = randsample([0,1],2*N,true);

modeltype = 'exp_prop';

regularization = 'false';
kernelfun = @ARD_kernelfun;
theta = [0,0];

f = @(x) prediction_bin(theta, xtrain_norm, ctrain, x, kernelfun, modeltype, [], regularization);

for i = 1:N
[mu_c(i), mu_y, sigma2_y, Sigma2_y, dmuc_dx(:,i),dmuy_dx(:,i), dsigma2y_dx(:,i), dSigma2y_dx(:,:,i)] =  prediction_bin(theta, xtrain_norm, ctrain, x(:,i), kernelfun, modeltype, [], regularization);
end

for i = 1:N
drdx(:,i)=test_matrix_deriv(f, x(:,i), 1e-12);
end

figure();
plot(dmuy_dx(:)); hold on
plot(drdx(:)); hold off;
sqrt(mse(dmuy_dx(:),drdx))

figure();
plot(dmuc_dx(:)); hold on
plot(drdx(:)); hold off;
sqrt(mse(dmuc_dx(:),drdx))


figure();
plot(dsigma2y_dx(:)); hold on
plot(drdx(:)); hold off;


figure();
plot(dsigma2y_dx(:)); hold on
plot(drdx(:)); hold off;

sqrt(mse(dsigma2y_dx(:),drdx))


f = @(x) ARD_kernelfun(theta, xtrain_norm, x, [], regularization);
% dC_dxnum = zeros(size(dC_dx)); %dC_dxnum = zeros([size(xtrain_norm, 2), size(x, 2)]);
dC_dxnum = zeros(40,20);
dC_dx= zeros(40,20);
for i = 1:N
[C, dC, dC_dx(:,i)] = ARD_kernelfun(theta, xtrain_norm, x(:,i), [], regularization);

    dC_dxnum(:,i)=test_matrix_deriv(f, x(:,i), 1e-9);
end
sqrt(mse(dC_dx(:), dC_dxnum(:)))
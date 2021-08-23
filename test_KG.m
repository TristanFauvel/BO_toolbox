
nx = 50;
ns = 10;
x_range = linspace(0,1,nx);
s_range = linspace(0,1,ns);
regularization = 'nugget';
post = [];

[p,q]= ndgrid(s_range, x_range);

xtest = [p(:),q(:)]';
c0 = [ctrain, 0];
c1 = [ctrain,1];
post =  prediction_bin(theta, xtrain_norm, ctrain, [], model, post);
[mu_c,  mu_y, sigma2_y, ] =  prediction_bin(theta, xtrain_norm, ctrain, xtest, model, post);
s0 = 0.5;
[c, g] =  prediction_bin(theta, xtrain_norm, ctrain, [s0*ones(1,nx); x_range], model, post);
[gmax, i] = max(g);
xmax = [s0; x_range(i)];
cmax = max(c);
N= size(mu_c,1);
U = zeros(1,N);
for i = 1:N
    [mu_c0,  mu_y0] =  prediction_bin(theta, [xtrain_norm, xtest(:,i)], c0, [s0*ones(1,nx); x_range], model, []);
    [mu_c1,  mu_y1] =  prediction_bin(theta, [xtrain_norm, xtest(:,i)], c1, [s0*ones(1,nx); x_range], model, []);
    u = (max(mu_y0)-gmax).*(1-mu_c(i)) + (max(mu_y1)-gmax).*mu_c(i);
%      u = (max(mu_c0)-cmax).*(1-mu_c(i)) + (max(mu_c1)-cmax).*mu_c(i);

    U(i) = u;
end


init_guess = [];
options.method = 'lbfgs';
options.verbose = 1;
ncandidates =5;
% s0 = rand_interval(model.lb_norm(1:ns), model.ub_norm(1:ns));
[xbest, ybest] = multistart_minConf(@(x)to_maximize_mean_bin_GP(theta, xtrain_norm, ctrain, x, model, post), [s0; model.lb_norm(1+model.ns:end)], [s0; model.ub_norm(1+model.ns:end)],ncandidates,[], []);
ybest = -ybest;

% xbest = xmax;
% ybest = gmax;
c0 = [ctrain, 0];
c1 = [ctrain,1];
u_est = zeros(size(U));
for i = 1:N
    u_est(i) =  knowledge_grad(theta, xtrain_norm, ctrain, xtest(:,i),model, post, c0, c1, xbest, ybest, [s0; model.lb_norm(1+model.ns:end)],[s0; model.ub_norm(1+model.ns:end)]);
end

[new_x, new_x_norm] = acquisition_fun(theta, xtrain_norm, ctrain,model, post, approximation);

figure();
subplot(1,2,1)
imagesc(s_range, x_range, reshape(U, [ns, nx])'); hold on
scatter(new_x_norm(1), new_x_norm(2), 15, 'w', 'filled'); hold off;
colorbar
set(gca,'Ydir', 'normal')
xlabel('context')
ylabel('variable')
subplot(1,2,2)
imagesc(s_range, x_range, reshape(-u_est,ns,nx)'); hold on;
scatter(new_x_norm(1), new_x_norm(2), 15, 'w', 'filled'); hold off;
set(gca,'Ydir', 'normal')
colorbar
xlabel('context')
ylabel('variable')





D = 2;
dUdx= zeros(N, D);
dUdxnum = zeros(N, D);
% %%%
for i = 1:N
    [~,dUdx(i,:)] = knowledge_grad(theta, xtrain_norm, ctrain, xtest(:,i),model, post, c0, c1, xbest, ybest, [s0; model.lb_norm(1+model.ns:end)],[s0; model.ub_norm(1+model.ns:end)]);
    dUdxnum(i,:) = test_matrix_deriv(@(xt)  knowledge_grad(theta, xtrain_norm, ctrain, xtest(:,i),model, post, c0, c1, xbest, ybest, [s0; model.lb_norm(1+model.ns:end)],[s0; model.ub_norm(1+model.ns:end)]), xtest(:,i), 1e-9);
    
end

mr = 2;
mc = 2;
figure()
subplot(mr, mc,1)
imagesc(reshape(dUdx(:,1),ns,nx)'); hold on;
set(gca,'Ydir', 'normal')
colorbar

subplot(mr, mc,2)
imagesc(reshape(dUdx(:,2),ns,nx)'); hold on;
set(gca,'Ydir', 'normal')
colorbar

subplot(mr, mc,3)
imagesc(reshape(dUdxnum(:,1),ns,nx)'); hold on;
set(gca,'Ydir', 'normal')
colorbar

subplot(mr, mc,4)
imagesc(reshape(dUdxnum(:,2),ns,nx)'); hold on;
set(gca,'Ydir', 'normal')
colorbar

figure()
plot(dUdx); hold on;
plot(dUdxnum); hold off;



%%
model.regularization = 'none';
dkdx = zeros(N,D);
dk_dxnum = zeros(N,D);
f = @(xt)  model.kernelfun(theta, xt, xt, false, model.regularization);
for i = 1:N
    [~, ~, ~, dkxxdx] = model.kernelfun(theta, xtest(:,i), xtest(:,i), false, model.regularization);
    dkdx(i,:)= dkxxdx;
    dk_dxnum(i,:) = test_matrix_deriv(f, xtest(:,i), 1e-9);
end
  
figure()
plot(dkdx(:)); hold on;
plot(dkdxnum(:)); hold off;

figure();
subplot(1,2,1)
imagesc(dkdx)
subplot(1,2,2)
imagesc(dk_dxnum)

sqrt(max((dk_dxnum(:)-dkdx(:)).^2))

N = 10;
xrange = linspace(0,1,N);
D = 2;
link = @logistic;
if D == 1
    x= xrange;
    theta= [0,0];
    
elseif D == 2
    [p,q] = meshgrid(xrange,xrange);
    x = [p(:), q(:)]';
    theta= [0,0,0];
    
end

N = size(x,2);
kernelfun = @ARD_kernelfun;
y = mvnrnd(zeros(1,size(x,2)), kernelfun(theta,x,x, 'true', 'nugget'));
ntr = 25;
itr = randsample(N,ntr);
xtrain = x(:,itr);
xtrain_norm = xtrain;
ctrain = link(y(itr))>rand(1,ntr);
c0 = [ctrain, 0];
c1 = [ctrain,1];
post =[];
model.modeltype= 'laplace';
model.D = D;

xt =rand(D,1);
lb_norm = zeros(D,1);
ub_norm = ones(D,1);

model.kernelfun = kernelfun;
model.lb_norm = lb_norm;
model.ub_norm = ub_norm;
model.regularization = 'nugget';
model.type = 'classification';
ncandidates = 5;
init_guess = [];
options.verbose= 1;
options.method = 'lbfgs';
model.link = link;
[xbest, ybest] = multistart_minConf(@(x)to_maximize_mean_bin_GP(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm, model.ub_norm,ncandidates,[], []);
ybest = -ybest;


dUdx = zeros(D,N);
dUdxnum= zeros(D,N);
U = zeros(1,N);
post =  prediction_bin(theta, xtrain_norm, ctrain, [], model, []);

% %%%
n= 1;
U = zeros(n,N);
dUdx= zeros(n,N, D);
dUdxnum = zeros(n,N, D);
% %%%
for i = 1:N
    [U(:,i),dUdx(:,i,:)] = knowledge_grad(theta, xtrain_norm, ctrain, x(:,i), model, post, c0, c1, xbest, ybest, lb_norm,ub_norm);
    dUdxnum(:,i,:) = test_matrix_deriv(@(xt) knowledge_grad(theta, xtrain_norm, ctrain, xt,model, post, c0, c1, xbest, ybest, lb_norm,ub_norm), x(:,i), 1e-9);
    
end

figure()
subplot(2,1,1)
plot(U);
subplot(2,1,2)
plot(dUdx); hold on
plot(dUdxnum); hold off

%%
k=2;
dUdx = squeeze(dUdx);
dUdxnum = squeeze(dUdxnum);
mr=2;
mc = 2;
figure()
subplot(mr,mc,1)
imagesc(reshape(U, N, N))
colorbar
set(gca, 'Ydir', 'normal')

subplot(mr,mc,2)
imagesc(reshape(dUdx(:,k), N, N))
colorbar
set(gca, 'Ydir', 'normal')

subplot(mr,mc,3)
imagesc(reshape(dUdxnum(:,k), N, N))
colorbar
set(gca, 'Ydir', 'normal')

%%
U = -U;
dUdx = -dUdx;
dUdxnum = -dUdxnum;

ncandidates = 10;
estimated_xmax  = multistart_minConf(@(x) knowledge_grad(theta, xtrain_norm, ctrain, x,model, post, c0, c1, xbest, ybest, lb_norm,ub_norm), lb_norm, ub_norm, ncandidates, init_guess, options);
[a,b] = max(U);
true_xmax = x(:,b);

if D == 1
    figure()
    subplot(1,2,1)
    plot(x, U); hold on;
    scatter(estimated_xmax, 0, 15, 'k', 'filled'); hold on;
    scatter(true_xmax, 0, 15, 'r', 'filled'); hold on;
    title('KG')
    subplot(1,2,2)
    plot(dUdx); hold on;
    plot(dUdxnum); hold on;
    title('dKGdx')
    legend('dKGdx', 'num dKGdx')
elseif D == 2
    figure()
    imagesc(xrange, xrange, reshape(U, sqrt(N), sqrt(N))); hold on;
    set(gca, 'Ydir', 'normal')
    title('KG')
    scatter(estimated_xmax(1), estimated_xmax(2), 15, 'k', 'filled'); hold on;
    scatter(true_xmax(1), true_xmax(2), 15, 'r', 'filled'); hold on;
    colorbar
    
    
    mr = 2;
    mc = 2;
    
    figure()
    subplot(mr,mc,1)
    imagesc(xrange, xrange, reshape(dUdx(1,:), sqrt(N), sqrt(N)))
    colorbar
    set(gca, 'Ydir', 'normal')
    title('dUdx')
    
    subplot(mr,mc,2)
    imagesc(xrange, xrange, reshape(dUdx(2,:), sqrt(N), sqrt(N)))
    colorbar
    set(gca, 'Ydir', 'normal')
    title('dUdx')
    
    subplot(mr,mc,3)
    imagesc(xrange, xrange, reshape(dUdxnum(1,:), sqrt(N), sqrt(N)))
    colorbar
    set(gca, 'Ydir', 'normal')
    title('dUdx num')
    
    subplot(mr,mc,4)
    imagesc(xrange, xrange, reshape(dUdxnum(2,:), sqrt(N), sqrt(N)))
    colorbar
    set(gca, 'Ydir', 'normal')
    title('dUdx num')
    
end
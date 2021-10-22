clear all
add_bo_module;
graphics_style_presentation;
 
close all
rng(1)

currentFile = mfilename( 'fullpath' );
[pathname,~,~] = fileparts( currentFile );
 figure_path = [pathname, '/Figures/'];
savefigures = 1;

link = @normcdf; %inverse link function

n = 50;
lb = 0; ub = 1;
x = linspace(lb, ub, n);
[p,q]= meshgrid(x);
x2d = [p(:), q(:)]';

n = 3000;
x = linspace(0,1, n);

%%
x2d = [x;0*ones(1,n)];

%%
d =1;
ntr = 50; %100

x0 =0;
% theta= [-1;1];
theta.cov= [4;-2];

modeltype = 'exp_prop'; % Approximation method
base_kernelfun =  @ARD_kernelfun;%kernel used within the preference learning kernel, for subject = computer
kernelname = 'ARD';

link = @normcdf; %inverse link function for the classification model
% model.regularization = 'nugget';
% model.base_kernelfun = base_kernelfun;
cond_base_kernelfun = @(theta, xi, xj, training, reg) conditioned_kernelfun(theta, base_kernelfun, xi, xj, training, x0,reg);

 kernelfun = @(theta, xi, xj, training, reg) conditional_preference_kernelfun(theta, base_kernelfun, xi, xj, training,reg,x0);

 
regularization = 'nugget';
meanfun = 0;
type = 'preference';
hyps.ncov_hyp =2; % number of hyperparameters for the covariance function
hyps.nmean_hyp =0; % number of hyperparameters for the mean function
hyps.hyp_lb = -10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
hyps.hyp_ub = 10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
D = 1;
 
model = gp_classification_model(D, meanfun, kernelfun, regularization, ...
    hyps, lb, ub, type, link, modeltype, kernelname, []);

post = [];
model.D = 1;
regularization = 'none';
g =mvnrnd(zeros(1,n),cond_base_kernelfun(theta.cov, x, x, 'false', regularization));


figure()
plot(normcdf(g-g(1)))

f = g-g';
f= f(:);

%%%%%%%%%%
f = g;

%%%%%%%%%%
model2 = model;
model2.type = 'classification';
model2.kernelfun = base_kernelfun;


D=1;
if strcmp(model.kernelname, 'Matern52') || strcmp(model.kernelname, 'Matern32') %|| strcmp(model.kernelname, 'ARD')
    approximation.method = 'RRGP';
else
    approximation.method = 'SSGP';
end
approximation.nfeatures = 4096;
approximation.decoupled_bases= 1;
[approximation.phi_pref, approximation.dphi_pref_dx, approximation.phi, approximation.dphi_dx]= sample_features_preference_GP(theta, D, model, approximation);

sample_g = sample_GP(theta,  x, g, model2, approximation);
% 
% figure();
% plot(g); hold on;
% plot(sample_g(x));

rng(1) %2
nsamp= ntr;
rd_idx = randsample(size(x2d,2), nsamp, 'true');
% rd_idx= 500*ones(1000,1);
xtrain= x2d(:,rd_idx);
ytrain= f(rd_idx)';
ctrain = link(ytrain)>rand(nsamp,1);

xtrain = xtrain(:,1:ntr);
ctrain = ctrain(1:ntr);
ytrain = ytrain(1:ntr);

post = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), [], post);

rng(1)
nsamples = 4;
KSSx = kernelselfsparring_tour(theta, xtrain(:,1:ntr), ctrain(1:ntr), model, post, approximation, nsamples);
MVTx = MVT(theta, xtrain(:,1:ntr), ctrain(1:ntr), model, post, approximation,nsamples);
x1 = MVTx(1);


[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), [x;x0*ones(1,n)], post);


%%
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), [x;x1*ones(1,n)], post);

mr = 1;
mc = 2;
i = 0;

Y = normcdf(mvnrnd(mu_y,Sigma2_y,5000));
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(1)]);
fig.Color =  background_color;
layout1 = tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');
i = 0;
Xlim = [0,1];

nexttile();
i=i+1;
[p1, p2] = plot_distro(x, mu_c, Y, C(3,:), C(1,:), linewidth); hold on;
p3 = plot(x, normcdf(g - sample_g(x1)),'color', C(2,:), 'linewidth', linewidth);
% scatter(xtrain(1,1:ntr), ctrain(1:ntr), 'filled');
xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$P(x>x_1)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
%title('Inferred value function $g(x)$','Fontsize', Fontsize)
box off
s1 = scatter(KSSx , normcdf(sample_g(KSSx)- sample_g(x1)), 10*markersize, C(1,:), '+','LineWidth',1.5); hold on;
s2 = scatter(MVTx , normcdf(sample_g(MVTx)- sample_g(x1)), 10*markersize, C(2,:), '+','LineWidth',1.5); hold on;

legend([p3, p2, p1, s1, s2], '$P(x>x_1)$', '$p(\Phi[f(x,x_1)]|\mathcal{D})$', '$\mu_c(x,x_1)$', 'KernelSelfSparring', 'MUC','NumColumns',2)
legend box off

model.nsamples = 25;
KSSx = kernelselfsparring_tour(theta, xtrain(:,1:ntr), ctrain(1:ntr), model, post, approximation);
MVTx = MVT(theta, xtrain(:,1:ntr), ctrain(1:ntr), model, post, approximation);

nexttile();
i=i+1;
[p1, p2] = plot_distro(x, mu_c, Y, C(3,:), C(1,:), linewidth); hold on;
p3 = plot(x, normcdf(g - sample_g(x1)),'color', C(2,:), 'linewidth', linewidth);
% scatter(xtrain(1,1:ntr), ctrain(1:ntr), 'filled');
xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$P(x>x_1)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
box off
s1 = scatter(KSSx , normcdf(sample_g(KSSx)- sample_g(x1)), 10*markersize, C(1,:), '+','LineWidth',1.5); hold on;
s2 = scatter(MVTx , normcdf(sample_g(MVTx)- sample_g(x1)), 10*markersize, C(2,:), '+','LineWidth',1.5); hold on;

legend([p3, p2, p1, s1, s2], '$P(x>x_1)$','$p(\Phi[f(x,x_1)]|\mathcal{D})$', '$\mu_c(x,x_1)$', 'KernelSelfSparring', 'MUC','NumColumns',2)
legend box off

darkBackground(fig,background,[1 1 1])

figname  = 'batch_PBO';
  export_fig(fig, [figure_path,'/' , figname, '.pdf']);
export_fig(fig, [figure_path,'/' , figname, '.png']);


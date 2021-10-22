clear all
close all;
rng(4) %3
n = 100;
lb = 0; ub =1;
x = linspace(lb, ub,n);
graphics_style_presentation;
 currentFile = mfilename( 'fullpath' );
[pathname,~,~] = fileparts( currentFile );
 figure_path = [pathname, '/Figures/'];
savefigures = 1;

% Prior mean of the gaussian process
regularization = 'nugget';
meanfun= @constant_mean;

kernelname = 'ARD';
kernelfun =@ARD_kernelfun;
theta.cov = [3,2];
theta.mean = 0;
Kard = kernelfun(theta.cov,x,x, 'true', regularization);
mu_y_ard = meanfun(x,theta.mean);
y = mvnrnd(mu_y_ard, Kard);

ntr = 5; 
i_tr= randsample(n,ntr);
% i_tr(3)=100 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ntr = 3; %%%%%%%%%%%
xtrain = x(:,i_tr);
ytrain = y(:, i_tr);

 
hyps.ncov_hyp =2; % number of hyperparameters for the covariance function
hyps.nmean_hyp =1; % number of hyperparameters for the mean function
hyps.hyp_lb = -10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
hyps.hyp_ub = 10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);

D = 1;
type = 'regression';
model = gp_regression_model(D, meanfun, kernelfun, regularization, hyps, lb, ub, kernelname);

[mu_y, sigma2_y,dmu_dx, sigma2_dx, Sigma2_y, dSigma2_dx, post] = model.prediction(theta, xtrain, ytrain, x, []); 
sample_post = mvnrnd(mu_y, Sigma2_y);

theta_w = theta;
theta_w.cov = [4,3];
[mu_y_w, sigma2_y_w,~,~, Sigma2_y_w] = model.prediction(theta_w, xtrain, ytrain, x, []);
sample_post_w = mvnrnd(mu_y_w, Sigma2_y_w);


hyp.mean = 0;
hyp.cov = theta.cov;
update = 'cov';
% ncov_hyp = 2;
% nmean_hyp = 1;
% update = 'cov';
% init_guess = hyp;
% hyp_lb = -10*ones(1,2);
% hyp_ub = 10*ones(1,2);
% options_theta.method = 'lbfgs';
% hyp.cov = multistart_minConf(@(hyp)minimize_negloglike(hyp, xtrain, ytrain, kernelfun, meanfun, ncov_hyp, nmean_hyp, update), [hyp_lb,0], [hyp_ub,0],10, [], options_theta);
hyp = model.model_selection(xtrain, ytrain, hyp, update);

 

[mu_y_ml, sigma2_y_ml,~,~, Sigma2_y_ml] = model.prediction(hyp, xtrain, ytrain, x, []);
 sample_post_ml = mvnrnd(mu_y_ml, Sigma2_y_ml);
 

mr = 1;
mc = 3;
legend_pos = [-0.18,1];

fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(1)]);
fig.Color =  background_color;
layout = tiledlayout(mr,mc, 'TileSpacing', 'compact', 'padding','compact');
i = 0;


nexttile();
i=i+1;
h1 = plot_gp(x,mu_y_w, sqrt(sigma2_y_w), C(1,:),linewidth, 'background', background);
h2 = plot(x, y, 'Color',  C(2,:),'LineWidth', linewidth); hold on;
% errorshaded(x,mu_y_w, sqrt(sigma2_y_w), 'Color',  C(1,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold on
%errorshaded(x,mu_y_w, sqrt(sigma2_y_w), 'Color',  C(1,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold on
h3 = plot(xtrain, ytrain, 'ro', 'MarkerSize', 10, 'color', C(2,:)); hold on;
scatter(xtrain, ytrain, 2*markersize, C(2,:), 'filled'); hold on;
%title('Posterior distribution with wrong hyperparameters','Fontsize',Fontsize, 'interpreter', 'latex')
box off
xlabel('$x$')
ylabel('$f(x)$')
yl = get(gca,'Ylim');
set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1], 'Ytick', floor([yl(1), 0, yl(2)]), 'Fontsize', Fontsize);
legend([h1, h2, h3], 'GP posterior, random $\theta$', 'True function', 'Data', 'Location', 'northoutside')
legend box off

nexttile();
i=i+1;
h1 = plot_gp(x,mu_y_ml, sqrt(sigma2_y_ml), C(1,:),linewidth, 'background', background);
plot(x, y, 'Color',  C(2,:),'LineWidth', linewidth); hold on;
% errorshaded(x,mu_y, sqrt(sigma2_y), 'Color',  C(1,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold on
h2 = plot(xtrain, ytrain, 'ro', 'MarkerSize', 10, 'color', C(2,:)); hold on;
scatter(xtrain, ytrain, 2*markersize, C(2,:), 'filled'); hold on;
ylabel('$f(x)$')
box off
%title('Posterior distribution','Fontsize',Fontsize, 'interpreter', 'latex')
xlabel('$x$')
set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1], 'Ylim', yl, 'Ytick', floor([yl(1), 0, yl(2)]), 'Fontsize', Fontsize');
legend(h1, 'GP posterior, learned $\theta$', 'Location', 'northoutside')
legend box off


nexttile();
i=i+1;
h1 = plot_gp(x,mu_y, sigma2_y, C(1,:),linewidth, 'background', background);
h2 = plot(x, y, 'Color',  C(2,:),'LineWidth', linewidth); hold on;
% errorshaded(x,mu_y, sqrt(sigma2_y), 'Color',  C(1,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold on
plot(xtrain, ytrain, 'ro', 'MarkerSize', 10, 'color', C(2,:)); hold on;
scatter(xtrain, ytrain, 2*markersize, C(2,:), 'filled'); hold on;
ylabel('$f(x)$')
box off
%title('Posterior distribution','Fontsize',Fontsize, 'interpreter', 'latex')
xlabel('$x$')
set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1], 'Ylim', yl, 'Ytick', floor([yl(1), 0, yl(2)]), 'Fontsize', Fontsize');
legend(h1, 'GP posterior, true $\theta$', 'Location', 'northoutside')
legend box off


colormap(cmap)
box off

darkBackground(fig,background,foreground)


figname  = 'GP_regression_hyps';
  export_fig(fig, [figure_path,'/' , figname, '.pdf']);
export_fig(fig, [figure_path,'/' , figname, '.png']);





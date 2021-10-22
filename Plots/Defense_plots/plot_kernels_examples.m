clear all
rng(2)
x = linspace(0,1,100);
graphics_style_presentation;
currentFile = mfilename( 'fullpath' );
[pathname,~,~] = fileparts( currentFile );
 figure_path = [pathname, '/Figures/'];
savefigures = 1;

% Prior mean of the gaussian process
meanfun= @constant_mean;
regularization = 'nugget';
theta.cov = [3,0];
theta.mean = 0;
Kard = ARD_kernelfun(theta.cov,x,x, true, regularization);
mu_y_ard = constant_mean(x,theta.mean);
sigma2_y_ard = diag(Kard);
 sample_ard = mvnrnd(mu_y_ard, Kard);

theta.cov = [1,0,-3];
Klin = linear_kernelfun(theta.cov,x,x, true, regularization);
mu_y_lin = constant_mean(x,theta.mean);
sigma2_y_lin = diag(Klin);
 sample_lin = mvnrnd(mu_y_lin, Klin);

theta.cov = [0,0,0.4];
Kper= periodic_kernelfun(theta.cov,x,x, true, regularization);
mu_y_per = constant_mean(x,theta.mean);
sigma2_y_per = diag(Kper);
sample_per = mvnrnd(mu_y_per, Kper);

mr = 2;
mc = 3;
legend_pos = [-0.18,1.15];

fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  background_color;
layout = tiledlayout(mr,mc, 'TileSpacing', 'compact', 'padding','compact');
i = 0;

nexttile();
i=i+1;
ax1 = imagesc(x, x, Kard); hold off;
xlabel('$x$','Fontsize',Fontsize)
ylabel('$x''$','Fontsize',Fontsize)
set(gca,'YDir','normal')
title({'Squared exponential kernel','$K(x,x'') = \sigma^{2}e^{\frac{\parallel(x-x'')\parallel^2}{2\lambda^2}}$'},'Fontsize',Fontsize, 'interpreter', 'latex')
pbaspect([1 1 1])
box off
set(gca,'XLim',[0,1],'YLim',[0,1],'YTick',[0,0.5,1], 'Xtick',[0,0.5, 1], 'Fontsize', Fontsize)
% cb = colorbar;
% cb_lim = get(cb, 'Ylim');
% cb_tick = get(cb, 'Ytick');
% set(cb,'Ylim', cb_lim, 'Ytick', [cb_tick(1),cb_tick(end)]);
% 

nexttile();
i=i+1;
ax1 = imagesc(x, x, Klin); hold off;
xlabel('$x$','Fontsize',Fontsize)
ylabel('$x''$','Fontsize',Fontsize)
title({'Linear kernel','$K(x,x'') = \sigma_{b}^{2}+\sigma^{2}(x-c)\left(x^{\prime}-c\right)$'},'Fontsize',Fontsize, 'interpreter', 'latex')
set(gca,'YDir','normal')
pbaspect([1 1 1])
box off
set(gca,'XLim',[0,1],'YLim',[0,1],'YTick',[0,0.5,1], 'Xtick',[0,0.5, 1], 'Fontsize', Fontsize)
% cb = colorbar;
% cb_lim = get(cb, 'Ylim');
% cb_tick = get(cb, 'Ytick');
% set(cb,'Ylim', cb_lim, 'Ytick', [cb_tick(1),cb_tick(end)]);


nexttile();
i=i+1;
ax1 = imagesc(x, x, Kper); hold off;
xlabel('$x$','Fontsize',Fontsize)
ylabel('$x''$','Fontsize',Fontsize)
title({'Periodic kernel','$K(x,x'') = \sigma^{2} \exp \left(-\frac{2 \sin ^{2}\left(\pi\left|x-x^{\prime}\right| / p\right)}{\lambda^{2}}\right)$'},'Fontsize',Fontsize, 'interpreter', 'latex')
set(gca,'YDir','normal')
pbaspect([1 1 1])
box off
set(gca,'XLim',[0,1],'YLim',[0,1],'YTick',[0,0.5,1], 'Xtick',[0,0.5, 1], 'Fontsize', Fontsize)
% cb = colorbar;
% cb_lim = get(cb, 'Ylim');
% cb_tick = get(cb, 'Ytick');
% set(cb,'Ylim', cb_lim, 'Ytick', [cb_tick(1),cb_tick(end)]);


% colorbar
colormap(cmap)

%% Now, I plot the prior distribution corresponding to each kernel, along with samples from this distribution

nexttile();
i=i+1;
% errorshaded(x,mu_y_ard, sqrt(sigma2_y_ard), 'Color',  C(end,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold on
h1 = plot_gp(x,mu_y_ard, sigma2_y_ard, C(1,:), linewidth, 'background', background); hold on
h2 = plot(x, sample_ard, 'Color',  w,'LineWidth', linewidth/2); hold off;
box off
xlabel('$x$')
ylabel('$f(x)$')
set(gca, 'Fontsize', Fontsize)

nexttile();
i=i+1;
% errorshaded(x,mu_y_lin, sqrt(sigma2_y_lin), 'Color',  C(end,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold on
h1 = plot_gp(x,mu_y_lin, sigma2_y_lin, C(1,:), linewidth, 'background', background); hold on
h2 =  plot(x, sample_lin, 'Color',  w,'LineWidth', linewidth/2); hold off;
box off
xlabel('$x$')
set(gca, 'Fontsize', Fontsize)


nexttile();
i=i+1;
% errorshaded(x,mu_y_per, sqrt(sigma2_y_per), 'Color',  C(end,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold on
h1 = plot_gp(x,mu_y_per, sigma2_y_per,  C(1,:), linewidth, 'background', background); hold on
h2 = plot(x, sample_per, 'Color',  w,'LineWidth', linewidth/2); hold off;
box off
xlabel('$x$')
set(gca, 'Fontsize', Fontsize)


darkBackground(fig,background,[1 1 1])

figname  = 'Kernels_comparison';
  export_fig(fig, [figure_path,'/' , figname, '.pdf']);
export_fig(fig, [figure_path,'/' , figname, '.png']);


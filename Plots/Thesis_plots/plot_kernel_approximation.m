letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
graphics_style_paper;
figure_path = '/home/tfauvel/Documents/PhD/Figures/Thesis_figures/Chapter_1/';

% kernelfun = @ARD_kernelfun;
% kernelname = 'ARD';
% method = 'RRGP'; %'RRGP'
% theta = [-2*log(0.1),0];
% lengthscale = exp(-theta(1)/2);
kernelfun = @Matern52_kernelfun;
kernelname = 'Matern52';
method = 'RRGP'; %'RRGP'
lengthscale = 1/10;
theta = [log(lengthscale),0];

D = 1;
n= 100;
x = linspace(0,1,n);

x0 = 0.5;

regularization = 'nugget';
K = kernelfun(theta,x0,x, true, regularization);

figure();
imagesc(kernelfun(theta,x,x, true, regularization));

approximation.nfeatures = m;
approximation.method = method;

phi = sample_features_GP(theta, model, approximation);
K1 = phi(x0)*phi(x)';

approximation.nfeatures  = 8;
phi = sample_features_GP(theta, D, kernelname,approximation);
K2 = phi(x0)*phi(x)';

approximation.nfeatures  = 16;
phi = sample_features_GP(theta, model, approximation);
K3 = phi(x0)*phi(x)';

approximation.nfeatures = 32;
phi = sample_features_GP(theta, model, approximation);
K4 = phi(x0)*phi(x)';

approximation.nfeatures = 64;
phi = sample_features_GP(theta, model, approximation);
K5 = phi(x0)*phi(x)';

mr = 1;
mc = 4;
legend_pos = [0.02,1];

fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  [1 1 1];
layout = tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');
i = 0;
% 
% nexttile();
% i=i+1;
% ax1 = imagesc(x, x, K); hold off;
% xlabel('$x$','Fontsize',Fontsize)
% ylabel('$x''$','Fontsize',Fontsize)
% title({'Squared exponential kernel','$K(x,x'') = \sigma^{2}e^{\frac{\parallel(x-x'')\parallel^2}{2\lambda^2}}$'},'Fontsize',Fontsize, 'interpreter', 'latex')
% set(gca,'YDir','normal')
% pbaspect([1 1 1])
% text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
% box off
% % colorbar
% colormap(cmap)

% nexttile();
% i=i+1;
% plot(x, K1, 'Color',  C(2,:),'LineWidth', linewidth/2); hold on;
% plot(x, K, 'Color',  C(1,:),'LineWidth', linewidth/2); hold off;
% box off
% xlabel('$x$')
% set(gca,'YLim',[-0.1,1],'YTick',[0,0.5,1])

% ylabel('$K(x_0,x)$')

ticks = [-0.5,0.5]./lengthscale;
% ylim = [0,1];
nexttile();
i=i+1;
plot(x, K2, 'Color',  C(2,:),'LineWidth', linewidth/2); hold on;
plot(x, K, 'Color',  C(1,:),'LineWidth', linewidth/2); hold off;
box off
xlabel('$x$')
ylim = get(gca, 'YLim');
set(gca,'YLim',ylim,'YTick',[0,0.5,1], 'Xtick',[0,0.5, 1])
set(gca,'xticklabel',{[num2str(ticks(1)),'$\lambda$'], '0', [num2str(ticks(2)),'$\lambda$']}, 'Fontsize', Fontsize)
% text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
title('$m = 8$', 'Fontsize', Fontsize)

nexttile();
i=i+1;
plot(x, K3, 'Color',  C(2,:),'LineWidth', linewidth/2); hold on;
plot(x, K, 'Color',  C(1,:),'LineWidth', linewidth/2); hold off;
box off
xlabel('$x$')
set(gca,'YLim',ylim,'YTick',[0,0.5,1], 'Xtick',[0,0.5, 1])
set(gca,'xticklabel',{[num2str(ticks(1)),'$\lambda$'], '0', [num2str(ticks(2)),'$\lambda$']}, 'Fontsize', Fontsize)
% text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
title('$m = 16$', 'Fontsize', Fontsize)

nexttile();
i=i+1;
plot(x, K4, 'Color',  C(2,:),'LineWidth', linewidth/2); hold on;
plot(x, K, 'Color',  C(1,:),'LineWidth', linewidth/2); hold off;
box off
xlabel('$x$')
set(gca,'YLim',ylim,'YTick',[0,0.5,1], 'Xtick',[0,0.5, 1])
set(gca,'xticklabel',{[num2str(ticks(1)),'$\lambda$'], '0', [num2str(ticks(2)),'$\lambda$']}, 'Fontsize', Fontsize)
% text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
title('$m = 32$', 'Fontsize', Fontsize)

nexttile();
i=i+1;
plot(x, K5, 'Color',  C(2,:),'LineWidth', linewidth/2); hold on;
plot(x, K, 'Color',  C(1,:),'LineWidth', linewidth/2); hold off;
box off
xlabel('$x$')
set(gca,'YLim',ylim,'YTick',[0,0.5,1], 'Xtick',[0,0.5, 1])
set(gca,'xticklabel',{[num2str(ticks(1)),'$\lambda$'], '0', [num2str(ticks(2)),'$\lambda$']}, 'Fontsize', Fontsize)
% text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
title('$m = 64$', 'Fontsize', Fontsize)


figname  = 'approximationimation';
folder = [figure_path,figname];
savefig(fig, [folder,'/', figname, '.fig'])
exportgraphics(fig, [folder,'/' , figname, '.pdf']);
exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);


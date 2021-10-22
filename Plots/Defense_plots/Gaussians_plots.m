n = 100;
lb = -3;
ub = 3;
x = linspace(lb,ub, n);
graphics_style_presentation;

sigma=1;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(1)]);
fig.Color =  background_color;
plot(x, normpdf(x), 'linewidth', linewidth, 'Color', C(1,:))
box off
darkBackground(fig, background, foreground)
ticks = [-1, 0, 1]./sigma;
% set(gca,'xtick', ticks, 'xticklabel',{['$\mu-$', num2str(abs(ticks(1))),'$\sigma$'], '\mu', ['\mu + ', num2str(ticks(3)),'$\sigma$']}, 'Fontsize', Fontsize)
set(gca,'xtick', ticks, 'xticklabel',{'$\mu-\sigma$', '\mu', '\mu + \sigma$'}, 'Fontsize', Fontsize)

 

[p,q] = meshgrid(x,x);
X= [p(:), q(:)];
mu = [0,0];
e = 0.3;
Sigma = [1-e,e;e,1+e];
G = mvnpdf(X, mu, Sigma);
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(1)]);
fig.Color =  background_color;
imagesc(x, x, reshape(G,n,n)); hold on;
xlabel('$x$','Fontsize', Fontsize)
ylabel('$x''$','Fontsize', Fontsize)
title('$P(x>x'')$','Fontsize', Fontsize)
set(gca,'YDir','normal')
% set(gca,'XTick',[0 0.5 1],'YTick',[0 0.5 1],'Fontsize', Fontsize)
pbaspect([1 1 1])
c = colorbar;
c.FontName = 'CMU Serif';
c.FontSize = Fontsize;
c.Limits = [0,1];
set(c, 'XTick', [0,1]);

RdBu=cbrewer('seq', 'Blues', 255, 'spline');
cmap = flipud(RdBu);

colormap(cmap)
box off
darkBackground(fig, background, foreground)

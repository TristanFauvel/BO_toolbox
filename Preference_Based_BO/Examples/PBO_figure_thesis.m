clear all
add_bo_module;
graphics_style_paper;
figure_path = '/home/tfauvel/Documents/PhD/Figures/Thesis_figures/Chapter_1/';

close all
rng(1)

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

link = @normcdf; %inverse link function

%% Define the range of parameters
n = 50;
x = linspace(0,1, n);
d =1;
ntr = 5;

[p,q]= meshgrid(x);
x2d = [p(:), q(:)]';

x0 = x(:,1);

modeltype = 'exp_prop'; % Approximation method
base_kernelfun =  @Matern52_kernelfun;%kernel used within the preference learning kernel, for subject = computer
base_kernelname = 'Matern52';
kernel_approximation = 'RRGP';
condition.x0 = x0;
condition.y0 = 0;

kernelfun = @(theta, xi, xj, training) conditional_preference_kernelfun(theta, base_kernelfun, xi, xj, training);
link = @normcdf; %inverse link function for the classification model

% gfunc = @(x) forretal08(x)/10;
% gfunc = @(x) normpdf(x, 0.5, 0.2);
% g = gfunc(x)-gfunc(x0);

theta= [-1;1];
g = mvnrnd(zeros(1,n),base_kernelfun(theta, x, x, 'false'));
g = g-g(1);

f = g'-g;
f= f(:);

nsamp= 500;
rd_idx = randsample(size(x2d,2), nsamp, 'true');
xtrain= x2d(:,rd_idx);
ytrain= f(rd_idx);
ctrain = link(ytrain)>rand(nsamp,1);


[mu_c,  mu_f, sigma2_f] = prediction_bin_preference(theta, xtrain(:,1:ntr), ctrain(1:ntr), x2d, kernelfun, 'modeltype', modeltype);

[~,  mu_g, sigma2_g, Sigma2_g, ~,~,~,~,~,~,post] = prediction_bin_preference(theta, xtrain(:,1:ntr), ctrain(1:ntr), [x; x0*ones(1,n^d)], kernelfun, 'modeltype', modeltype);
mu_g = -mu_g; %(because prediction_bin_preference considers P(x1 > x2);
    


%% Find the true global optimum of g
[gmax, id_xmax] = max(g);
xmax = x(id_xmax);
mr = 2;
mc = 3;

fig=figure('units','centimeters','outerposition',1+[0 0 16 height(mr)]);
fig.Color =  [1 1 1];

i = 0;
tiledlayout(mr,mc, 'TileSpacing' , 'tight', 'Padding', 'tight')

% nexttile([1,2])
nexttile
i=i+1;
imagesc(x, x, reshape(link(f),n,n)); hold on;
xlabel('$x$')
ylabel('$x''$')
title('$P(x''>x)$')
set(gca,'YDir','normal')
set(gca,'XTick',[0 0.5 1],'YTick',[0 0.5 1])
pbaspect([1 1 1])
c = colorbar;
c.Limits = [0,1];
set(c, 'XTick', [0,1]);
colormap(cmap)
text(-0.25,1.1,['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)

% nexttile([1,2])
nexttile
i=i+1;
imagesc(x, x, reshape(mu_c, n,n)); hold on;
scatter(xtrain(1, ctrain(1:ntr)==1),xtrain(2, ctrain(1:ntr)==1), markersize, 'o', 'k','filled'); hold on;
scatter(xtrain(1, ctrain(1:ntr)==0),xtrain(2, ctrain(1:ntr)==0), markersize, 'o','k'); hold off;
xlabel('$x$')
ylabel('$x''$')
title('$P(x''>x | \mathcal{D})$')
set(gca,'YDir','normal')
set(gca,'XTick',[0 0.5 1],'YTick',[0 0.5 1])
pbaspect([1 1 1])
c = colorbar;
c.Limits = [0,1];
set(c, 'XTick', [0,1]);
colormap(cmap)
text(-0.25,1.1,['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)

% nexttile([1,2])
nexttile
i=i+1;
imagesc(x, x, reshape(mu_f, n,n)); hold on;
scatter(xtrain(1, ctrain(1:ntr)==1),xtrain(2, ctrain(1:ntr)==1), markersize, 'o', 'k','filled'); hold on;
scatter(xtrain(1, ctrain(1:ntr)==0),xtrain(2, ctrain(1:ntr)==0), markersize, 'o','k'); hold off;
xlabel('$x$')
ylabel('$x''$')
title('$\mu_f(x,x'')$')
set(gca,'YDir','normal')
set(gca,'XTick',[0 0.5 1],'YTick',[0 0.5 1])
pbaspect([1 1 1])
c = colorbar;
text(-0.25,1.1,['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
colormap(cmap)

% nexttile([1,2])
nexttile
i=i+1;
plot_gp(x, mu_g, sigma2_g, C(1,:), linewidth); hold on;
plot(x,g, 'Color',  C(1,:),'linewidth',linewidth); hold on
xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$g(x)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1])
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3))
%title('Inferred value function $g(x)$','Fontsize', Fontsize)
text(-0.25,1.15,['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
box off

%% Plot a sample
decoupled_bases = 1;
nfeatures = 256;
[sample_f, sample_g]= sample_preference_GP(x, theta, xtrain(:,1:ntr), ctrain(1:ntr), base_kernelname, kernelfun, modeltype, kernel_approximation, decoupled_bases, base_kernelfun, nfeatures, condition, post);



% nexttile([1,2])
nexttile
i=i+1;
plot(x, sample_g,'k','linewidth',linewidth); hold on;
[sample_gmax, id] = max(sample_g);
xmax = x(id);
v1=vline(xmax, 'Color', 'k','LineWidth',linewidth,'ymax',sample_gmax, 'LineStyle', '-'); hold off;
xlabel('$x$')
ylabel('$\tilde{g}(x)$')
set(gca,'XTick',[0 0.5 1])
%title('Sample from $P(g|\mathcal{D})$')
text(-0.25,1.15,['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
box off

%% Plot the variance of the duels (to explain the principle of MaximumVarianceChallenge)
% nexttile([1,2])
nexttile
i=i+1;
[~,~,~,~,~,~,~,~, var_muc, dvar_muc_dx] =  prediction_bin_preference(theta, xtrain(:,1:ntr), ctrain(1:ntr), [x;xmax*ones(size(x))], kernelfun, 'modeltype', modeltype);
d = size(x,1);
plot(x, var_muc,'k','linewidth',linewidth);
xlabel('$x$')
ylabel('$V(\mu_c|\mathcal{D})$')
box off;
% pbaspect([1 1 1])
% title('Variance of the expected outcome')
text(-0.25,1.15,['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)



figname  = 'PBO';
folder = [figure_path,figname];
savefig(fig, [folder,'/', figname, '.fig'])
exportgraphics(fig, [folder,'/' , figname, '.pdf']);
exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);


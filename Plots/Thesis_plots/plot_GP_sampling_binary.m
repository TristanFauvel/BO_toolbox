%% Sample GP

clear all
close all
add_gp_module;
figure_path = '/home/tfauvel/Documents/PhD/Figures/Thesis_figures/Chapter_1/';

rng(13);%12
n=100; %resolution
x = linspace(0,1,n);

gen_kernelfun = @Matern52_kernelfun;
kernelfun = @Matern52_kernelfun;
kernelname = 'Matern52';
approximation.method = 'RRGP';
link = @normcdf;

modeltype = 'exp_prop';
post = [];
regularization = 'nugget';

model.regularization = regularization;
model.kernelfun = kernelfun;
model.link = link;
model.modeltype = modeltype;
model.kernelname = kernelname;
model.D = 1;
% gen_kernelfun = @ARD_kernelfun;
% kernelfun = @ARD_kernelfun;
% kernelname = 'ARD';
% approximation.method = 'SSGP';
%
%
theta.cov = [log(1/10),0];
theta_gen.cov = theta.cov;
theta.mean= 0;

Sigma =gen_kernelfun(theta_gen.cov, x,x, true, 'nugget');

g =  mvnrnd(constant_mean(x,0), gen_kernelfun(theta_gen.cov, x,x, true, 'nugget')); %generate a function


figure(); plot(link(g))


graphics_style_paper;

% h=figure(1);
% h.Color =  [1 1 1];
% h.Name = 'Value function';
% plot(x, g, 'Color', colo(end, :),'LineWidth',1.5); hold on;
% plot(x, y, 'Color', 'k','LineWidth',1.5); hold off;
% xlabel('x','Fontsize',Fontsize)
% ylabel('f(x)','Fontsize',Fontsize)
% box off


N=5;
% idx_data = randsample(n,N);
% idx_data= sort(idx_data);
idx_data = 50:70;
N = numel(idx_data);
x_data = x(idx_data);
y_data = g(idx_data)';
c_data = link(y_data)>rand(N,1);
hyp = theta.cov;

m=10000;
fx = NaN(m, n);

hyps.ncov_hyp =2; % number of hyperparameters for the covariance function
hyps.nmean_hyp =0; % number of hyperparameters for the mean function
hyps.hyp_lb = -10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
hyps.hyp_ub = 10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
D = 1;
meanfun = 0;
type = 'classification';
model = gp_classification_model(D, meanfun, kernelfun, regularization, hyps, type, link, modeltype);

[mu_c,  mu_f, sigma2_f, Sigma2_f, ~, ~,~,~,var_muc, ~, post] = model.prediction(hyp, x_data, c_data, x, post);

D= 1;
nfeatures = 256*2;
approximation.decoupled_bases = 1;
approximation.nfeatures = nfeatures;

for i =1:m
    [gs, dgsdx, decomposition]= sample_binary_GP(theta.cov, x_data, c_data, model, approximation, post);
    fx(i, :)=gs(x);
end

W = Wasserstein2(mu_f, Sigma2_f, fx);

fig=figure('units','centimeters','outerposition',1+[0 0 16 1.3/2*16]);
fig.Color =  [1 1 1];
subplot(2,1,1)
errorshaded(x,mu_c, sqrt(var_muc), 'Color',  cmap(11,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold on
box off;
subplot(2,1,2)
errorshaded(x,mean(link(fx),1), std(link(fx),1), 'Color',  cmap(11,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold on
box off;

fig=figure('units','centimeters','outerposition',1+[0 0 16 1.3/2*16]);
fig.Color =  [1 1 1];
subplot(2,1,1)
plot(x,mu_c,'LineWidth', linewidth); hold on
plot(x, mean(link(fx),1),'LineWidth', linewidth); hold off
box off;
subplot(2,1,2)
plot(x,sqrt(var_muc),'LineWidth', linewidth); hold on
plot(x,std(link(fx),1),'LineWidth', linewidth); hold off
box off;

sqrt(var_muc)
mr = 2;
mc = 4;
legend_pos = [-0.18,1.0];
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  [1 1 1];
layout = tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');
i = 0;

nexttile();
i=i+1;
plot(x,g, 'Color',  cmap(end,:),'LineWidth', linewidth); hold on;
errorshaded(x,mu_f, sqrt(sigma2_f), 'Color',  cmap(11,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold on
plot(x_data, y_data, 'ro', 'MarkerSize', 10, 'color', cmap(end,:)); hold on;
scatter(x_data, y_data, markersize, cmap(end,:), 'filled'); hold off;
set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1]);
xlabel('$x$')
box off
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
set(gca, 'Ylim', [-2,2],'Fontsize', Fontsize);

mcol = parula;
nexttile();
i=i+1;
plot(x, decomposition.sample_prior(x), 'Color',  mcol(1,:),'LineWidth', linewidth); hold on ;
plot(x, decomposition.update(x), 'Color', mcol(end,:),'LineWidth', linewidth); hold on ;
plot(x, gs(x), 'Color', mcol(floor(256/3),:),'LineWidth', linewidth); hold off ;
legend('Sample from the prior', 'Update', 'Posterior')
box off;
legend boxoff
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1],'Fontsize', Fontsize);
xlabel('$x$')

nexttile();
i=i+1;
plot(x , fx(1:5,:)','LineWidth', linewidth/2); hold on;
box off
%title('Samples from the posterior distribution')
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1],'Fontsize', Fontsize);
xlabel('$x$')

nexttile();
i=i+1;
errorshaded(x, mean(fx,1), sqrt(var(fx,1)), 'Color',  cmap(11,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold on
box off
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1],'Fontsize', Fontsize);
xlabel('$x$')
set(gca, 'Ylim', [-2,2]);


nexttile();
i=i+1;
plot(x, mean(fx,1), 'Color',  cmap(1,:),'LineWidth', linewidth); hold on;
plot(x, mu_f, 'Color',  cmap(end,:),'LineWidth', linewidth); hold off
% legend('sample mean', 'posterior mean')
box off
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1],'Fontsize', Fontsize);
xlabel('$x$')

nexttile();
i=i+1;
plot(x, var(fx,1), 'Color',  cmap(1,:),'LineWidth', linewidth); hold on;
plot(x, sigma2_f, 'Color',  cmap(end,:),'LineWidth', linewidth); hold off
% legend('sample variance', 'posterior variance')
box off
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1],'Fontsize', Fontsize);
xlabel('$x$')

crosscov= cov(fx);
cl = [min([Sigma2_f(:);crosscov(:)]),max([Sigma2_f(:);crosscov(:)])];
nexttile();
i=i+1;
ax1 = imagesc(x,x,crosscov,cl);
title('Sample covariance')
pbaspect([1,1,1])
%colorbar
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
colormap(cmap)
box off
xlabel('$x$','Fontsize',Fontsize)
ylabel('$x''$','Fontsize',Fontsize)
set(gca,'YDir','normal','Fontsize', Fontsize)


nexttile();
i=i+1;
ax2 = imagesc(x,x,Sigma2_f,cl);
pbaspect([1,1,1])
title('True covariance')
%colorbar
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
colormap(cmap)
box off
xlabel('$x$','Fontsize',Fontsize)
ylabel('$x''$','Fontsize',Fontsize)
set(gca,'YDir','normal')


figname  = 'GP_sampling_binary';
folder = [figure_path,figname];
savefig(fig, [folder,'/', figname, '.fig'])
exportgraphics(fig, [folder,'/' , figname, '.pdf']);
exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);



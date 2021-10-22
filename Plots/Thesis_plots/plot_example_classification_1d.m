%% Gaussian process classification

clear all;
close all;
add_gp_module;

figure_path = '/home/tfauvel/Documents/PhD/Figures/Thesis_figures/Chapter_1/';
n=100;

rng(3)
graphics_style_paper;
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

colo= othercolor('GnBu7');
modeltype = 'exp_prop'; % or 'laplace'

link = @normcdf;

kernelname = 'ARD';
kernelfun = @ARD_kernelfun; 
theta_true= [3;1.2];

% Generate a function 
lb = 0;
ub = 1;

x = linspace(0,1,n);
y = mvnrnd(constant_mean(x,0), kernelfun(theta_true, x,x, 'false', 'false')); 
y=y-mean(y);

p= link(y);


figure();
subplot(1,2,1);
plot(x,y);
subplot(1,2,2);
plot(x,p);

ntr =10; % Number of data points

i_tr= randsample(n,ntr,'true');
xtrain = x(:,i_tr);
y_tr = y(:, i_tr);
ctrain = p(i_tr)>rand(1,ntr);

x_test = x;
y_test = y;

% GP classification with the correct hyperparameters
theta.cov =theta_true ; % rand(size(theta_true));
theta.mean = 0;
post = [];
D = 1;
meanfun = 0;
regularization = 'nugget';
type = 'classification';
hyps.ncov_hyp =2; % number of hyperparameters for the covariance function
hyps.nmean_hyp =0; % number of hyperparameters for the mean function
hyps.hyp_lb = -10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
hyps.hyp_ub = 10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
D = 1;
condition = [];

model = gp_classification_model(D, meanfun, kernelfun, regularization, hyps, lb, ub, type, link, modeltype, kernelname, condition);

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc, dvar_muc_dx]= model.prediction(theta, xtrain, ctrain, x_test,post);
fun = @(x_test) model.prediction(theta, xtrain, ctrain, x_test, post);
dF = test_matrix_deriv(fun, x_test, 1e-8);

%Xlim= [min(x),max(x)];
%Ylim = [-5,5];

legend_pos = [-0.1,1];

mr = 1;
mc = 2;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  background_color;
layout = tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');

i = 0;

Y = normcdf(mvnrnd(mu_y, Sigma2_y, 10000));


% nexttile();
% i=i+1;
% plot(x_test, mu_c, 'LineWidth',linewidth,'Color', C(1,:)) ; hold on;
% plot(x_test, p, 'LineWidth',linewidth,'Color',  C(2,:)) ; hold on;
% scatter(xtrain(ctrain == 1), ctrain(ctrain == 1), markersize, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k') ; hold on;
% scatter(xtrain(ctrain == 0), ctrain(ctrain == 0), markersize, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k') ; hold off;
% legend('$P(c=1)$', '$\mu_c(x)$','$c=1$', '$c=0$','Fontsize',Fontsize, 'Location', 'northeast')
% xlabel('$x$','Fontsize',Fontsize)
% legend boxoff
% grid off
% box off
% set(gca, 'Fontsize', Fontsize);
% text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
% ylabel('$P(c=1)$')

nexttile();
i=i+1;
[p1, p2] = plot_distro(x, mu_c, Y, C(3,:), C(1,:),linewidth);
p3 = plot(x_test, p, 'LineWidth',linewidth,'Color',  C(2,:)) ; hold on;
p4 = scatter(xtrain(ctrain == 1), ctrain(ctrain == 1), markersize, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k') ; hold on;
p5 = scatter(xtrain(ctrain == 0), ctrain(ctrain == 0), markersize, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k') ; hold off;
% legend('$P(c=1)$', '$\mu_c(x)$','$c=1$', '$c=0$','Fontsize',Fontsize, 'Location', 'northeast')
legend([ p3,p2, p1, p4, p5], '$P(c=1)$', 'Posterior', '$\mu_c(x)$','$c=1$', '$c=0$','Fontsize',Fontsize, 'Location', 'northeast')
xlabel('$x$','Fontsize',Fontsize)
legend boxoff
grid off
box off
% pbaspect([1 1 1])
set(gca, 'Fontsize', Fontsize);
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
ylabel('$P(c=1)$')


nexttile();
i=i+1;
p1 = plot_gp(x,mu_y, sigma2_y, C(1,:),linewidth);
p2 = plot(x,y,'LineWidth',linewidth,'Color', C(2,:)); hold off;
% errorshaded(x,mu_y, sqrt(sigma2_y), 'Color',  C(1,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold off
% legend('True function', 'Inferred function','Fontsize',Fontsize)
legend([p1, p2], {'Posterior GP','$f(x)$'},'Fontsize',Fontsize, 'Location', 'northeast')
xlabel('$x$','Fontsize',Fontsize)
legend boxoff
grid off
box off
% pbaspect([1 1 1])
set(gca, 'Fontsize', Fontsize);
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
ylabel('$f(x)$')

% nexttile();
% i=i+1;
% ax1 = imagesc(x, x, Sigma2_y); hold on;
% xlabel('$x$','Fontsize',Fontsize)
% ylabel('$x''$','Fontsize',Fontsize)
% %title('Posterior covariance','Fontsize',Fontsize, 'interpreter', 'latex')
% set(gca,'YDir','normal')
% pbaspect([1 1 1])
% cb = colorbar;
% cb_lim = get(cb, 'Ylim');
% cb_tick = get(cb, 'Ytick');
% set(cb,'Ylim', cb_lim, 'Ytick', [cb_tick(1),cb_tick(end)]);
% colormap(cmap)
% set(gca, 'Fontsize', Fontsize);
% text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
% title('Cov$(f(x),f(x'')|\mathcal{D})$')
% 
figname  = 'GP_classification';
folder = [figure_path,figname];
savefig(fig, [folder,'/', figname, '.fig'])
exportgraphics(fig, [folder,'/' , figname, '.pdf']);
exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);

 

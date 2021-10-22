clear all
add_gp_module
close all
rng(1)
graphics_style_presentation;

currentFile = mfilename( 'fullpath' );
[pathname,~,~] = fileparts( currentFile );
figure_path = [pathname, '/Figures/'];
savefigures = 1;

link = @normcdf; %inverse link function
regularization = 'nugget';
post = [];
%% Define the range of parameters
n = 80;
x = linspace(0,1, n);
d =1;

[p,q]= meshgrid(x);
x2d = [p(:), q(:)]';


modeltype = 'exp_prop'; % Approximation method
original_kernelfun =  @Matern52_kernelfun;%kernel used within the preference learning kernel, for subject = computer
x0 = 0;

kernelname = 'Matern52';


approximationimation = 'RRGP';

base_kernelfun = @(theta, xi, xj, training, reg) conditioned_kernelfun(theta, original_kernelfun, xi, xj, training, x0, reg);
% kernelfun = @(theta, xi, xj, training) preference_kernelfun(theta, base_kernelfun, xi, xj, training);
kernelfun= @(theta, xi, xj, training, reg) conditional_preference_kernelfun(theta, original_kernelfun, xi, xj, training, reg,x0);

link = @normcdf; %inverse link function for the classification model

% gfunc = @(x) forretal08(x)/10;
% gfunc = @(x) normpdf(x, 0.5, 0.2);
% g = gfunc(x)-gfunc(x0);

theta.mean = 0;
theta.cov= [-1;1];
g = mvnrnd(zeros(1,n),base_kernelfun(theta.cov, x, x, 'false', 'no'));
%g = g-g(1);

f = g-g';
f= f(:);
ntr = 10;
rd_idx = randsample(size(x2d,2), ntr, 'true');
rd_idx(end) = size(x2d,2);
xtrain= x2d(:,rd_idx);
ytrain= f(rd_idx);
ctrain = link(ytrain)>rand(ntr,1);

D = 1;
meanfun = 0;
type = 'classification';
hyps.ncov_hyp =2; % number of hyperparameters for the covariance function
hyps.nmean_hyp =1; % number of hyperparameters for the mean function
hyps.hyp_lb = -10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
hyps.hyp_ub = 10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
lb = 0;
ub = 1;

condition.x0 = 0;
condition.y0 = 0;
model = gp_classification_model(D, meanfun, kernelfun, regularization, hyps, lb,ub, type, link, modeltype, kernelname, condition);


[mu_c_cond,  mu_f_cond, sigma2_f_cond] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), x2d, post);
[mu_c_cond_x0,  mu_g_cond, sigma2_g_cond, Sigma2_g_cond, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), [x; x0*ones(d,n^d)], post);


%% Find the true global optimum of g
% [gmax, id_xmax] = max(g);
% xmax = x(id_xmax);
mr= 1;

fig=figure('units','centimeters','outerposition',1+[0 0 fwidth 1.1*fheight(mr)]);
fig.Color =  background_color;


cl = [0,1];
nexttile
i=i+1;
imagesc(x, x, reshape(link(f),n,n), cl); hold on;
xlabel('$x$','Fontsize', Fontsize)
ylabel('$x''$','Fontsize', Fontsize)
title('$P(x>x'')$','Fontsize', Fontsize)
set(gca,'YDir','normal')
set(gca,'XTick',[0 0.5 1],'YTick',[0 0.5 1],'Fontsize', Fontsize)
pbaspect([1 1 1])
c= colorbar;
c.FontName = 'CMU Serif';
c.FontSize = Fontsize;
c.Limits = [0,1];
set(c, 'XTick', [0,1]);
colormap(cmap)
p1 = scatter(xtrain(1, ctrain(1:ntr)==1),xtrain(2, ctrain(1:ntr)==1), markersize, 'MarkerFaceColor', k, 'MarkerEdgeColor', w); hold on;
p2 = scatter(xtrain(1, ctrain(1:ntr)==0),xtrain(2, ctrain(1:ntr)==0), markersize, 'MarkerFaceColor', w, 'MarkerEdgeColor', k); hold off;
legend([p1, p2], '$c=1$', '$c=0$','Fontsize',Fontsize, 'Location', 'northeast')
legend box off

darkBackground(fig,background,[1 1 1])
figname  = 'preference_function';
export_fig(fig, [figure_path,'/' , figname, '.pdf']);
export_fig(fig, [figure_path,'/' , figname, '.png']);

%%

fig=figure('units','centimeters','outerposition',1+[0 0 fwidth 1.1*fheight(mr)]);
fig.Color =  background_color;

h1 = plot_gp(x, mu_g_cond, sigma2_g_cond, C(1,:), linewidth, 'background', background); hold on;
h2 = plot(x, g, 'color', C(2,:), 'linewidth', linewidth);  hold on;

[max_y,idx] = max(mu_g_cond);
max_x = x(idx);
x1 = max_x;

xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$g(x)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
%title('Inferred value function $g(x)$','Fontsize', Fontsize)
box off
% pbaspect([1 1 1])
vline(max_x,'Linewidth',linewidth, 'ymax', max_y,  'LineStyle', '--', ...
    'Linewidth', 1, 'Color', k); hold off;
[xt,b] = sort([0,x1 1]);
xticks(xt)
lgs = {'0', '$x_1$', '1'};
xticklabels(lgs(b))
legend([h1 h2], 'Posterior GP : $p(g|\mathcal{D})$','True value function $g(x)$')
legend box off

darkBackground(fig,background,[1 1 1])
figname  = 'value_function';
export_fig(fig, [figure_path,'/' , figname, '.pdf']);
export_fig(fig, [figure_path,'/' , figname, '.png']);

%%

fig=figure('units','centimeters','outerposition',1+[0 0 fwidth 1.1*fheight(mr)]);
fig.Color =  background_color;

[mu_c_cond_x0,  mu_g_cond, sigma2_g_cond, Sigma2_g_cond, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), [x; max_x*ones(d,n^d)], post);

Y = normcdf(mvnrnd(mu_g_cond,Sigma2_g_cond,10000));

[p1, p2] = plot_distro(x, mu_c_cond_x0, Y, C(3,:), C(1,:), linewidth, 'background', background); hold on;
p3 = plot(x, normcdf(g-g(idx)), 'color', C(2,:), 'linewidth', linewidth);  hold off;

xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$P(c=1|x,x_1)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
box off
% pbaspect([1 1 1])
legend([p3, p2, p1], '$P(x>x_1)$', '$p(\Phi[f(x,x_1)]|\mathcal{D})$', '$\mu_c(x, x_1)$')
legend box off
[xt,b] = sort([0,x1, 1]);
xticks(xt)
lgs = {'0', '$x_1$', '1'};
xticklabels(lgs(b))

darkBackground(fig,background,[1 1 1])
figname  = 'MUC_1';
export_fig(fig, [figure_path,'/' , figname, '.pdf']);
export_fig(fig, [figure_path,'/' , figname, '.png']);


%%

fig=figure('units','centimeters','outerposition',1+[0 0 fwidth 1.1*fheight(mr)]);
fig.Color =  background_color;
figname  = 'MUC';
export_fig(fig, [figure_path,'/' , figname, '.pdf']);
export_fig(fig, [figure_path,'/' , figname, '.png']);
[max_y,b] = max(var_muc);
max_x = x(b);

p1= plot(x, var_muc, 'color', C(1,:), 'linewidth', linewidth);  hold on;
p2 = plot(x, mu_c_cond_x0.*(1-mu_c_cond_x0), 'color', C(2,:), 'linewidth', linewidth);  hold on;

box off
xlabel('$x$', 'Fontsize', Fontsize)
%  ylabel('V$[\Phi(f(x,x_1))|\mathcal{D}]$', 'Fontsize', Fontsize)
ylabel('Uncertainty', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
vline(max_x,'Linewidth',linewidth, 'ymax', max_y,  'LineStyle', '--', ...
    'Linewidth', 1,'Color', k); hold off;
x2 = max_x;
[xt,b] = sort([0,x1, x2, 1]);
xticks(xt)
lgs = {'0', '$x_1$', '$x_2$','1'};
xticklabels(lgs(b))
% legend([p1, p2], 'Epistemic uncertainty V$[\Phi(f(x,x_1))|\mathcal{D}]$', 'Aleatoric uncertainty V$(c| x, x_1, \mathcal{D})$')
legend([p1, p2], 'V$[\Phi(f(x,x_1))|\mathcal{D}]$', 'V$(c| x, x_1, \mathcal{D})$')

legend box off

darkBackground(fig,background,[1 1 1])
figname  = 'MUC_2';
export_fig(fig, [figure_path,'/' , figname, '.pdf']);
export_fig(fig, [figure_path,'/' , figname, '.png']);


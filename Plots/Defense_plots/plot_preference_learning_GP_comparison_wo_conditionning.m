clear all
add_gp_module
 close all
rng(1)
graphics_style_presentation;
currentFile = mfilename( 'fullpath' );
[pathname,~,~] = fileparts( currentFile );
 figure_path = [pathname, '/Figures/'];
savefigures = 1;

rng(4)
link = @normcdf; %inverse link function
regularization = 'nugget';
%% Define the range of parameters
n = 50;
D = 1;
ub = 1;
lb = 0;
x = linspace(lb, ub, n);
d =1;

[p,q]= meshgrid(x);
x2d = [p(:), q(:)]';


modeltype = 'exp_prop'; % Approximation method
x0 = 0.1;
condition.x0 = x0;

base_kernelname = 'Matern52';
original_kernelfun =  @Matern52_kernelfun;%kernel used within the preference learning kernel, for subject = computer
theta.cov= [-1;1];

base_kernelname = 'ARD_kernelfun';
original_kernelfun =  @ARD_kernelfun;%kernel used within the preference learning kernel, for subject = computer
theta.cov= [3;0];
theta.mean = 0;

approximationimation = 'RRGP';

base_kernelfun = @(theta, xi, xj, training, reg) conditioned_kernelfun(theta, original_kernelfun, xi, xj, training, x0, reg);
kernelfun_cond= @(theta, xi, xj, training, reg) conditional_preference_kernelfun(theta, original_kernelfun, xi, xj, training, reg,x0);
kernelfun= @(theta, xi, xj, training, reg) preference_kernelfun(theta, original_kernelfun, xi, xj, training, reg);
meanfun = 0;
type = 'preference';
hyps.ncov_hyp =2; % number of hyperparameters for the covariance function
hyps.nmean_hyp =0; % number of hyperparameters for the mean function
hyps.hyp_lb = -10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
hyps.hyp_ub = 10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);

model = gp_classification_model(D, meanfun, kernelfun, regularization, hyps, lb, ub, type, link, modeltype, base_kernelname, []);

model_cond = gp_classification_model(D, meanfun, kernelfun_cond, regularization, hyps, lb, ub, type, link, modeltype, base_kernelname, condition);

% gfunc = @(x) forretal08(x)/10;
% gfunc = @(x) normpdf(x, 0.5, 0.2);
% g = gfunc(x)-gfunc(x0);

g = mvnrnd(zeros(1,n),base_kernelfun(theta.cov, x, x, 'false', 'no'));
% g = g-g(1);

f = g-g';
f= f(:);


ntr =5;
rd_idx = randsample(size(x2d,2), ntr, 'true');
xtrain= x2d(:,rd_idx);
ytrain= f(rd_idx);
ctrain = link(ytrain)>rand(ntr,1);


[mu_c,  mu_f, sigma2_f,~, ~, ~, ~, ~, var_muc_f] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), x2d,  []);
[mu_c_x0,  mu_g, sigma2_g, Sigma2_g, ~, ~, ~, ~, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), [x; x0*ones(1,n^d)],  []);

[mu_c_cond,  mu_f_cond, sigma2_f_cond,~, ~, ~, ~, ~, var_muc_fcond] = model_cond.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), x2d, []);
[mu_c_cond_x0,  mu_g_cond, sigma2_g_cond, Sigma2_g_cond, ~, ~, ~, ~, var_muc_cond] = model_cond.prediction(theta, ...
    xtrain(:,1:ntr), ctrain(1:ntr), [x; x0*ones(d,n^d)],  []);


%% Find the true global optimum of g
[gmax, id_xmax] = max(g);
xmax = x(id_xmax);
% legend_pos = [-0.2,1];
legend_pos = [0.1,1];
nyticks =5;
mr = 2;
mc = 4;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth 0.8*fheight(mr)]);
fig.Color =  background_color;
tiledlayout(mr,mc, 'TileSpacing' , 'compact', 'Padding', 'compact')
i = 1;
nexttile(i)
plot(x, mu_g,'color', C(2,:), 'linewidth', linewidth); hold on;
plot(x, mu_g_cond, 'color', C(1,:), 'linewidth', linewidth); hold off;
xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$\mu_g(x)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
box off

legend('Without conditioning', ['Conditioning  on $g(x_0) = 0$, $x_0=$', num2str(x0)], 'location', 'northoutside')
legend box off

i = 2;
nexttile(i)
plot(x, sigma2_g,'color', C(2,:), 'linewidth', linewidth); hold on;
plot(x, sigma2_g_cond, 'color', C(1,:), 'linewidth', linewidth); hold off;
xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$\sigma^2_g(x)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
box off

i = 3;
nexttile(i)
plot(x, mu_c_x0,'color', C(2,:), 'linewidth', linewidth); hold on;
plot(x, mu_c_cond_x0, 'color', C(1,:), 'linewidth', linewidth); hold off;
xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$\mu_c(x,x_0)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
box off

i = 4;
nexttile(i)
plot(x, var_muc,'color', C(2,:), 'linewidth', linewidth); hold on;
plot(x, var_muc_cond, 'color', C(1,:), 'linewidth', linewidth); hold off;
xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$V[\Phi(f(x, x_0))]$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
box off

ntr = 100;
rd_idx = randsample(size(x2d,2), ntr, 'true');
xtrain= x2d(:,rd_idx);
ytrain= f(rd_idx);
ctrain = link(ytrain)>rand(ntr,1);


[mu_c,  mu_f, sigma2_f,~, ~, ~, ~, ~, var_muc_f] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), x2d,  []);
[mu_c_x0,  mu_g, sigma2_g, Sigma2_g, ~, ~, ~, ~, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), [x; x0*ones(1,n^d)],  []);

[mu_c_cond,  mu_f_cond, sigma2_f_cond,~, ~, ~, ~, ~, var_muc_fcond] = model_cond.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), x2d, []);
[mu_c_cond_x0,  mu_g_cond, sigma2_g_cond, Sigma2_g_cond, ~, ~, ~, ~, var_muc_cond] = model_cond.prediction(theta, ...
    xtrain(:,1:ntr), ctrain(1:ntr), [x; x0*ones(d,n^d)],  []);

i = 5;
nexttile(i)
plot(x, mu_g,'color', C(2,:), 'linewidth', linewidth); hold on;
plot(x, mu_g_cond, 'color', C(1,:), 'linewidth', linewidth); hold off;
xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$\mu_g(x)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
box off


i = 6;
nexttile(i)
plot(x, sigma2_g,'color', C(2,:), 'linewidth', linewidth); hold on;
plot(x, sigma2_g_cond, 'color', C(1,:), 'linewidth', linewidth); hold off;
xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$\sigma^2_g(x)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
box off

i = 7;
nexttile(i)
plot(x, mu_c_x0,'color', C(2,:), 'linewidth', linewidth); hold on;
plot(x, mu_c_cond_x0, 'color', C(1,:), 'linewidth', linewidth); hold off;
xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$\mu_c(x,x_0)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
box off

i = 8;
nexttile(i)
plot(x, var_muc,'color', C(2,:), 'linewidth', linewidth); hold on;
plot(x, var_muc_cond, 'color', C(1,:), 'linewidth', linewidth); hold off;
xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$V[\Phi(f(x, x_0))]$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
box off

darkBackground(fig,background,[1 1 1])

figname  = 'preference_learning_GP_comparison';
  export_fig(fig, [figure_path,'/' , figname, '.pdf']);
export_fig(fig, [figure_path,'/' , figname, '.png']);



%%
%Compute the 2-Wasserstein distance between conditionned vs non
%conditionned/
nreps = 32;
ntrains = 2:20:1000;
dist = zeros(nreps, numel(ntrains));
for j =1:nreps
    rng(j)
    g = mvnrnd(zeros(1,n),base_kernelfun(theta.cov, x, x, 'false', 'no'));
    f = g-g';
    f= f(:);
    for i = 1:numel(ntrains)
        ntr =ntrains(i);
        rd_idx = randsample(size(x2d,2), ntr, 'true');
        xtrain= x2d(:,rd_idx);
        ytrain= f(rd_idx);
        ctrain = link(ytrain)>rand(ntr,1);
        
        [mu_c_x0,  mu_g, sigma2_g, Sigma2_g, ~, ~, ~, ~, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), [x; x0*ones(1,n^d)], []);
        [mu_c_cond_x0,  mu_g_cond, sigma2_g_cond, Sigma2_g_cond, ~, ~, ~, ~, var_muc_cond] = model_cond.prediction(theta, ...
            xtrain(:,1:ntr), ctrain(1:ntr), [x; x0*ones(d,n^d)], []);
        dist(j,i) = Wasserstein2(mu_g, Sigma2_g, mu_g_cond, Sigma2_g_cond);
    end
end

mr = 1;
mc = 1;
legend_pos = [-0.18,1];

fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(1)]);
fig.Color =  background_color;
options.handle = fig;
options.alpha = 0.2;
options.error= 'sem';
options.line_width = linewidth;
options.semilogy = false;
options.cmap = C;
colors = colororder;
options.colors = C;
options.color_area = C(1,:);
options.color_line = C(1,:);
options.x_axis = ntrains;
plots =  plot_areaerrorbar(log10(real(dist)), options);
box off
ylabel('log$_{10}$(2-Wasserstein)')
xlabel('Size of the training set')

% legends = {'Decoupled-bases approx., RRGP', 'Decoupled-bases approx., SSGP', 'Weight-space approx., RRGP'};
% legend(plots, legends, 'Fontsize', Fontsize);

box off
% legend box off
xticklabs = ntrains
xticklabels(xticklabs)

set(gca, 'Fontsize', Fontsize)

pbaspect([2,1,1])
darkBackground(fig,background,[1 1 1])


figname  = 'Wasserstein_condition';
  export_fig(fig, [figure_path,'/' , figname, '.pdf']);
export_fig(fig, [figure_path,'/' , figname, '.png']);




%%

mr = 1;
mc = 2;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  background_color;

tiledlayout(mr,mc, 'TileSpacing' , 'compact', 'Padding', 'compact')
plot(x, mu_c_x0,'color', C(2,:), 'linewidth', linewidth); hold on;
plot(x, mu_c_cond_x0,'color', C(1,:), 'linewidth', linewidth); hold on;
plot(x, normcdf(g),'color', k, 'linewidth', linewidth); hold off;
xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$g(x)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)

ylim = get(gca, 'Ylim');
box off


%
% mr = 3;
% mc = 2;
% fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
% fig.Color =  background_color;
% tiledlayout(mr,mc, 'TileSpacing' , 'compact', 'Padding', 'compact')
% nexttile()
% ax1 = imagesc(x, x, reshape(mu_c, n,n)); hold on;
% xlabel('$x$','Fontsize',Fontsize)
% ylabel('$x''$','Fontsize',Fontsize)
% title('$\mu_c(x)$','Fontsize',Fontsize, 'interpreter', 'latex')
% set(gca,'YDir','normal')
% pbaspect([1 1 1])
% set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1],'Ylim', [0,1], 'Ytick', [0,0.5,1], 'Fontsize', Fontsize');
% text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
% cb = colorbar;
% cb_lim = get(cb, 'Ylim');
% cb_tick = get(cb, 'Ytick');
% set(cb,'Ylim', cb_lim, 'Ytick', [cb_tick(1),cb_tick(end)]);
% colormap(cmap)
% box off
%
% nexttile()
% ax1 = imagesc(x, x, reshape(mu_c_cond, n,n)); hold on;
% xlabel('$x$','Fontsize',Fontsize)
% ylabel('$x''$','Fontsize',Fontsize)
% title('$\mu_c(x)$','Fontsize',Fontsize, 'interpreter', 'latex')
% set(gca,'YDir','normal')
% pbaspect([1 1 1])
% set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1],'Ylim', [0,1], 'Ytick', [0,0.5,1], 'Fontsize', Fontsize');
% text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
% cb = colorbar;
% cb_lim = get(cb, 'Ylim');
% cb_tick = get(cb, 'Ytick');
% set(cb,'Ylim', cb_lim, 'Ytick', [cb_tick(1),cb_tick(end)]);
% colormap(cmap)
% box off
%
% nexttile()
% ax1 = imagesc(x, x, reshape(sigma2_f, n,n)); hold on;
% xlabel('$x$','Fontsize',Fontsize)
% ylabel('$x''$','Fontsize',Fontsize)
% title('Cov$(f(x),f(x'')|\mathcal{D})$','Fontsize',Fontsize, 'interpreter', 'latex')
% set(gca,'YDir','normal')
% pbaspect([1 1 1])
% set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1],'Ylim', [0,1], 'Ytick', [0,0.5,1], 'Fontsize', Fontsize');
% text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
% cb = colorbar;
% cb_lim = get(cb, 'Ylim');
% cb_tick = get(cb, 'Ytick');
% set(cb,'Ylim', cb_lim, 'Ytick', [cb_tick(1),cb_tick(end)]);
% colormap(cmap)
% box off
%
% nexttile()
% ax1 = imagesc(x, x, reshape(sigma2_f_cond, n,n)); hold on;
% xlabel('$x$','Fontsize',Fontsize)
% ylabel('$x''$','Fontsize',Fontsize)
% title('Cov$(f(x),f(x'')|\mathcal{D})$','Fontsize',Fontsize, 'interpreter', 'latex')
% set(gca,'YDir','normal')
% pbaspect([1 1 1])
% set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1],'Ylim', [0,1], 'Ytick', [0,0.5,1], 'Fontsize', Fontsize');
% text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
% cb = colorbar;
% cb_lim = get(cb, 'Ylim');
% cb_tick = get(cb, 'Ytick');
% set(cb,'Ylim', cb_lim, 'Ytick', [cb_tick(1),cb_tick(end)]);
% colormap(cmap)
% box off
%
%
%
% nexttile()
% ax1 = imagesc(x, x, reshape(var_muc_f, n,n)); hold on;
% xlabel('$x$','Fontsize',Fontsize)
% ylabel('$x''$','Fontsize',Fontsize)
% title('Cov$(f(x),f(x'')|\mathcal{D})$','Fontsize',Fontsize, 'interpreter', 'latex')
% set(gca,'YDir','normal')
% pbaspect([1 1 1])
% set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1],'Ylim', [0,1], 'Ytick', [0,0.5,1], 'Fontsize', Fontsize');
% text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
% cb = colorbar;
% cb_lim = get(cb, 'Ylim');
% cb_tick = get(cb, 'Ytick');
% set(cb,'Ylim', cb_lim, 'Ytick', [cb_tick(1),cb_tick(end)]);
% colormap(cmap)
% box off
%
% nexttile()
% ax1 = imagesc(x, x, reshape(var_muc_fcond, n,n)); hold on;
% xlabel('$x$','Fontsize',Fontsize)
% ylabel('$x''$','Fontsize',Fontsize)
% title('Cov$(f(x),f(x'')|\mathcal{D})$','Fontsize',Fontsize, 'interpreter', 'latex')
% set(gca,'YDir','normal')
% pbaspect([1 1 1])
% set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1],'Ylim', [0,1], 'Ytick', [0,0.5,1], 'Fontsize', Fontsize');
% text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
% cb = colorbar;
% cb_lim = get(cb, 'Ylim');
% cb_tick = get(cb, 'Ytick');
% set(cb,'Ylim', cb_lim, 'Ytick', [cb_tick(1),cb_tick(end)]);
% colormap(cmap)
% box off

%%



acquisition_funs = {'active_sampling_binary'};
acquisition_fun = @active_sampling_binary;
acquisition_name = 'BALD';
maxiter = 80;%total number of iterations

nreplicates = 2; %20;
nreps= nreplicates;
nacq = numel(acquisition_funs);


% wbar = waitbar(0,'Computing...');
rescaling = 0;
if rescaling ==0
    load('benchmarks_table.mat')
else
    load('benchmarks_table_rescaled.mat')
end
objectives = benchmarks_table.fName; %; 'Ursem_waves';'forretal08'; 'camel6';'goldpr'; 'grlee12';'forretal08'};
nobj =numel(objectives);
seeds = 1:nreplicates;
update_period = maxiter+2;
conditions = [0,1];

cum_regret_maxvar= NaN(nreps,maxiter+1);
cum_regret_rand= NaN(nreps,maxiter+1);
link = @normcdf;
modeltype = 'exp_prop';
theta = [1,-1]';

D = 1;
kernelname = 'ARD';
model.base_kernelfun = @ARD_kernelfun;
model.kernelname = kernelname;
model.modeltype = modeltype;
model.link = link;
model.regularization = 'nugget';
model.lb = zeros(D,1);
model.ub = ones(D,1);
model.lb_norm = zeros(D,1);
model.ub_norm = ones(D,1);
model.type  = 'preference';
model.D = D;
for c = 1:2
    clear('xtrain', 'xtrain_norm', 'ctrain', 'score');
    condition = conditions(c);
    for r=1:nreplicates  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        seed  = seeds(r);
        objective = GPnd(D, theta, kernelname,seed);
        g = @(x) objective.do_eval(x);
        %             waitbar(((a-1)*nreplicates+r)/(nreplicates*nacq),wbar,'Computing...');
        [xtrain{r}, xtrain_norm{r}, ctrain{r}, score{r}] =  AL_preference_loop(acquisition_fun, seed, maxiter, theta, g, update_period, model,condition);
    end
    
    if c ==1
        score_wo = score;
    else
        score_w = score;
    end
end

scores{1} = cell2mat(score_wo');
scores{2} = cell2mat(score_w');
legends = {'Without conditionning', 'With conditionning'};

fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  background_color;
tiledlayout(mr, mc, 'TileSpacing', 'compact', 'padding','compact');
options.handle = fig;
options.alpha = 0.2;
options.error= 'sem';
options.line_width = linewidth/2;
options.semilogy = false;
options.cmap = C;
options.colors = colororder;
plots =  plot_areaerrorbar_grouped(scores, options);
legend(plots, legends, 'Fontsize', Fontsize);


clear all
add_bo_module
graphics_style_paper
figure_path = '/home/tfauvel/Documents/PhD/Figures/Thesis_figures/Chapter_1/';
close all


letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
legend_pos = [-0.05,1];

%% Define the range of parameters
n = 100;
x = linspace(0,1, n);


meanfun= @constant_mean;
nmean_hyp = 1;
theta.mean = zeros(nmean_hyp,1);

wrong_theta.mean = theta.mean;


%generate a function
D = 2;
objective = GPnd(D);
g = @(x) objective.do_eval(x);
theta.cov = objective.theta;
kernelfun = objective.kernelfun;
kernelname = objective.kernelname;
lb = objective.xbounds(:,1);
ub = objective.xbounds(:,2);

wrong_theta.cov =[2,7]; %[10,7];



nopt =3; %set nopt to maxiter +1 to get random acquisition
maxiter = 30;

nreps = 10;
cum_regret_true_hyp= NaN(nreps,maxiter+1);
cum_regret_wrong_hyp = NaN(nreps,maxiter+1);
cum_regret_learned_hyp = NaN(nreps,maxiter+1);

 theta_lb = -10*ones(size(theta.cov));
 theta_ub = 10*ones(size(theta.cov));
 max_g = 0
 lb_norm = zeros(D,1);
 ub_norm = ones(D,1);
for k = 1:nreps
    rng(k)
    ninit = maxiter + 1;
    acquisition_fun = @EI;
    [xtrain{k}, xtrain_norm{k}, ytrain{k}, score{k}]= BO_loop(g, maxiter, nopt, kernelfun, meanfun, theta, acquisition_fun, ninit, ub, lb, theta_lb, theta_ub, max_g, kernelname, lb_norm, ub_norm)
    [xtrain{k}, xtrain_norm{k}, ytrain{k}, score{k}]= BO_loop(g, maxiter, nopt, kernelfun, meanfun, theta, acquisition_fun, ninit, ub, lb, theta_lb, theta_ub, max_g, kernelname, lb_norm, ub_norm)
    ninit = 5;
    [xtrain{k}, xtrain_norm{k}, ytrain{k}, score{k}]= BO_loop(g, maxiter, nopt, kernelfun, meanfun, theta, acquisition_fun, ninit, ub, lb, theta_lb, theta_ub, max_g, kernelname, lb_norm, ub_norm)
end


mr = 1;
mc = 3;
fig=figure('units','centimeters','outerposition',1+[0 0 width height(mr)]);
fig.Color =  [1 1 1];
layout1 = tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');
i = 0;
Xlim = [0,1];
nexttile(layout1, 1, [1,2]);
i=i+1;
plot(x, y, 'Color',  cmap(end,:),'LineWidth', linewidth); hold on;
plot(x, yw, 'Color',  cmap(1,:),'LineWidth', linewidth); hold off;
legend('True $\theta$', 'Wrong $\theta$')
legend boxoff
xlabel('$x$')
% ylabel('$f(x)$')
set(gca, 'Fontsize', Fontsize, 'Xlim', Xlim); %,  'Ylim',Ylim)
grid off
box off
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
set(gca, 'Xtick', [0,1])


nexttile(layout1);
i=i+1;

options.handle = fig;
options.alpha = 0.2;
options.error= 'sem'; %std
options.line_width = linewidth;
options.color_area = cmap(1,:);
options.color_line = cmap(1,:);
h1 = plot_areaerrorbar(cum_regret_true_hyp, options); hold on;
options.color_area = cmap(220,:);
options.color_line = cmap(220,:);
h2 = plot_areaerrorbar(cum_regret_wrong_hyp, options); hold on;
options.color_area = cmap(180,:);
options.color_line = cmap(180,:);
h3 = plot_areaerrorbar(cum_regret_learned_hyp, options); hold on;

legend([h1 h2 h3], 'True $\theta$', 'Wrong $\theta$', 'Learned $\theta$', 'Location', 'northwest');
%legend('EI', 'Random')
xlabel('Iteration','Fontsize',Fontsize)
ylabel('Cumulative regret','Fontsize',Fontsize)
set(gca, 'Fontsize', Fontsize, 'Xlim', [0, maxiter])
grid off
box off
legend boxoff
set(gca, 'Xlim', [1, maxiter])


figname  = 'Bayesian_optimization_hyperparameters';
folder = [figure_path,figname];
savefig(fig, [folder,'\', figname, '.fig'])
exportgraphics(fig, [folder,'\' , figname, '.pdf']);
exportgraphics(fig, [folder,'\' , figname, '.png'], 'Resolution', 300);


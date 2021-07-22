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
true_theta.mean = zeros(nmean_hyp,1);
wrong_theta.mean = true_theta.mean;


%generate a function
D = 6;
kernelname = 'ARD';
theta = [1,-1,3,6,-5,2,3]';
objective = GPnd(D, theta, kernelname);
g = @(x) objective.do_eval(x);
true_theta.cov = objective.theta;
ncov_hyp = numel(true_theta.cov);
kernelfun = objective.kernelfun;
kernelname = objective.kernelname;
lb = objective.xbounds(:,1);
ub = objective.xbounds(:,2);

wrong_theta.cov =[10;7;6;3;-9;-4;0]; %[10,7];



nopt =3; %set nopt to maxiter +1 to get random acquisition
maxiter = 50;

nreps = 20;
cum_regret_true_hyp= NaN(nreps,maxiter+1);
cum_regret_wrong_hyp = NaN(nreps,maxiter+1);
cum_regret_learned_hyp = NaN(nreps,maxiter+1);

 theta_lb = -10*ones(ncov_hyp+nmean_hyp,1);
 theta_ub = 10*ones(ncov_hyp+nmean_hyp,1);
 max_g = 0;
 lb_norm = zeros(D,1);
 ub_norm = ones(D,1);
 acquisition_fun = @EI;
 
for k = 1:nreps
    
    ninit = maxiter + 1;
    theta= true_theta;
    seed = k;
    [~, ~, ~, true_theta_score{k}]= BO_loop(g, maxiter, nopt, kernelfun, meanfun, theta, acquisition_fun, ninit, ub, lb, theta_lb, theta_ub, max_g, kernelname, lb_norm, ub_norm, seed)
    theta= wrong_theta;
    [~, ~, ~, wrong_theta_score{k}]= BO_loop(g, maxiter, nopt, kernelfun, meanfun, theta, acquisition_fun, ninit, ub, lb, theta_lb, theta_ub, max_g, kernelname, lb_norm, ub_norm,seed)
    ninit = nopt;
    [~, ~, ~, learned_theta_score{k}]= BO_loop(g, maxiter, nopt, kernelfun, meanfun, theta, acquisition_fun, ninit, ub, lb, theta_lb, theta_ub, max_g, kernelname, lb_norm, ub_norm,seed)
end


mr = 1;
mc = 1;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  [1 1 1];
layout1 = tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');
options.handle = fig;
options.alpha = 0.2;
options.error= 'sem'; %std
options.line_width = linewidth;
options.color_area = C(1,:);
options.color_line = C(1,:);
h1 = plot_areaerrorbar(cell2mat(true_theta_score'), options); hold on;
options.color_area = C(2,:);
options.color_line = C(2,:);
h2 = plot_areaerrorbar(cell2mat(wrong_theta_score'), options); hold on;
options.color_area = C(3,:);
options.color_line = C(3,:);
h3 = plot_areaerrorbar(cell2mat(learned_theta_score'), options); hold on;

legend([h1 h2 h3], 'True $\theta$', 'Wrong $\theta$', 'Learned $\theta$', 'Location', 'northwest');
%legend('EI', 'Random')
xlabel('Iteration','Fontsize',Fontsize)
ylabel('$f(x^{\star\star})$','Fontsize',Fontsize)
set(gca, 'Fontsize', Fontsize, 'Xlim', [0, maxiter])
grid off
box off
legend boxoff
set(gca, 'Xlim', [1, maxiter])


figname  = 'Bayesian_optimization_hyperparameters';
folder = [figure_path,figname];
savefig(fig, [folder,'/', figname, '.fig'])
exportgraphics(fig, [folder,'/' , figname, '.pdf']);
exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);


clear all
add_bo_module
graphics_style_paper
figure_path = '/home/tfauvel/Documents/PhD/Figures/Thesis_figures/Chapter_1/';
close all


letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
legend_pos = [-0.05,1];

%% Define the range of parameters
n = 100;
lb = 0;
ub = 1;
x = linspace(lb, ub , n);


meanfun= @constant_mean;
nmean_hyp = 1;
true_theta.mean = zeros(nmean_hyp,1);
wrong_theta.mean = true_theta.mean;


%generate a function
D = 6;
kernelname = 'ARD';
theta.cov = [1,-1,3,6,-5,2,3]';
objective = GPnd(D, theta, kernelname);
g = @(x) objective.do_eval(x);
true_theta.cov = objective.theta.cov;
ncov_hyp = numel(true_theta.cov);
kernelfun = objective.kernelfun;
kernelname = objective.kernelname;
lb = objective.xbounds(:,1);
ub = objective.xbounds(:,2);

wrong_theta.cov =[10;7;6;3;-9;-4;0]; %[10,7];



regularization = 'nugget';
hyps.ncov_hyp =numel(theta.cov); % number of hyperparameters for the covariance function
hyps.nmean_hyp =0; % number of hyperparameters for the mean function
hyps.hyp_lb = -10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
hyps.hyp_ub = 10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
D = 6;
link = @normcdf;
model = gp_regression_model(D, meanfun, kernelfun, regularization, hyps, lb, ub, kernelname);



nopt =3; %set nopt to maxiter +1 to get random acquisition
maxiter = 50;

nreps = 20;
cum_regret_true_hyp= NaN(nreps,maxiter+1);
cum_regret_wrong_hyp = NaN(nreps,maxiter+1);
cum_regret_learned_hyp = NaN(nreps,maxiter+1);

 max_g = 0;
 lb_norm = zeros(D,1);
 ub_norm = ones(D,1);
 acquisition_fun = @EI;
 
for k = 1:nreps
    
    ninit = maxiter + 1;
    theta= true_theta;
    seed = k;
    [~, ~, ~, true_theta_score{k}]= BO_loop(g, maxiter, nopt, model, theta, acquisition_fun, ninit, max_g, seed);
    theta= wrong_theta;
    [~, ~, ~, wrong_theta_score{k}]= BO_loop(g, maxiter, nopt, model, theta, acquisition_fun, ninit, max_g, seed);
    ninit = nopt;
    [~, ~, ~, learned_theta_score{k}, theta_evo]= BO_loop(g, maxiter, nopt, model, theta, acquisition_fun, ninit, max_g, seed);
end
%[xtrain, xtrain_norm, ytrain, score, cum_regret]= BO_loop(g, maxiter, nopt, kernelfun, meanfun, theta, acquisition_fun, ninit, max_x, min_x, hyp_lb, hyp_ub, max_g, kernelname, lb_norm, ub_norm, seed)


mr = 1;
mc = 3;
x0 = [x; zeros(D-1,n)];
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  [1 1 1];
layout1 = tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');
nexttile()
plot(x, g(x0), 'color', C(1,:), 'linewidth', linewidth); hold on;
plot(x, mvnrnd(zeros(1,n), kernelfun(wrong_theta.cov, x0,x0, true, 'none')), 'color', C(2,:), 'linewidth', linewidth); hold on;
plot(x, mvnrnd(zeros(1,n), kernelfun(theta_evo(:,end), x0,x0, true, 'none')), 'color', C(3,:), 'linewidth', linewidth); hold off;
box off
xlabel('$x$')
ylabel('$f(x)$')
%legend('True function', 'Sample from the GP with wrong hyperparameters', 'Sample from the GP with learned hyperparameters')
set(gca, 'Fontsize',Fontsize)


nexttile([1 2])
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
set(gca, 'Xlim', [1, maxiter],'Fontsize',Fontsize)


figname  = 'Bayesian_optimization_hyperparameters';
folder = [figure_path,figname];
savefig(fig, [folder,'/', figname, '.fig'])
exportgraphics(fig, [folder,'/' , figname, '.pdf']);
exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);


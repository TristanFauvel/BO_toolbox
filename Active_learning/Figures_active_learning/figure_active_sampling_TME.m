
clear all
close all

add_bo_module;
graphics_style_paper
figure_path = '/home/tfauvel/Documents/PhD/Figures/Thesis_figures/Chapter_1/';

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
legend_pos = [-0.07,1];

regularization = 'nugget';
post = [];
n=100;
rng(2)

% kernelname = 'linear';
% kernelfun = @linear_kernelfun;
% theta_true = [5;0.5;0];
kernelname = 'ARD';
kernelfun = @ARD_kernelfun;
theta_true = [4;1];

theta= theta_true;


meanfun= @constant_mean;

%x = linspace(0,1, n);
x = [linspace(0,0.2, 20), linspace(0.4,0.6, 20), linspace(0.8,1,20)];
n = numel(x);

a= 7;
b = -3;
% g = @(x) normcdf(a*x+b);
%figure(); plot(g(x))

y = normcdf(mvnrnd(zeros(size(x)), kernelfun(theta_true, x,x, true, regularization))); %generate a function
figure(); plot(x, y)


xtest = x;

maxiter=4;
nopt =0;

xtrain = [];
ctrain = [];
options_theta.method = 'lbfgs';
options_theta.verbose = 1;

max_x = 1;
min_x = 0;
lb_norm = 0;
ub_norm = 1;
modeltype = 'exp_prop';
% new_x = random_acquisition_binary(theta, xtrain, ctrain, kernelfun, kernelname, modeltype,max_x, min_x, lb_norm, ub_norm);
idx = randsample(n,1);
new_x = x(idx);
theta_lb =-15*ones(size(theta));
theta_ub = 15*ones(size(theta));
lb = min_x;
ub = max_x;

for i =1:maxiter
    xtrain = [xtrain, new_x];
    %     ctrain = [ctrain, g(new_x)>0.5];
    ctrain = [ctrain, y(idx)>rand];     
    idx = randsample(n,1);
    new_x = x(idx);
    
end

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc]= prediction_bin(theta, xtrain, ctrain, xtest, model, post);
[new_x, ~, idx, L] = BALD_grid(x, theta, xtrain, ctrain,model, post);
[new_x, ~, idx, TME] = TME_sampling_binary(x, theta, xtrain, ctrain,model, post);

maxiter = 30;

nreps = 20; %50
cum_regret_maxvar= NaN(nreps,maxiter+1);
cum_regret_rand= NaN(nreps,maxiter+1);
cum_regret_BALD= NaN(nreps,maxiter+1);
cum_regret_TME= NaN(nreps,maxiter+1);
score_maxvar= NaN(nreps,maxiter);
score_rand= NaN(nreps,maxiter);
score_BALD= NaN(nreps,maxiter);
score_TME= NaN(nreps,maxiter);

ninit = maxiter+2;
for s = 1:nreps
seed = s;
[~,~, cum_regret_TME(s,:), score_TME(s,:)]= AL_loop_binary_grid(x, y, maxiter, nopt, kernelfun, theta, @TME_sampling_binary, ninit, theta_lb, theta_ub, lb, ub, seed);
[~,~, cum_regret_maxvar(s,:),score_maxvar(s,:)]= AL_loop_binary_grid(x, y, maxiter, nopt, kernelfun, theta, @maxvar_binary_grid, ninit, theta_lb, theta_ub, lb, ub, seed);
[~,~, cum_regret_rand(s,:),score_rand(s,:)]= AL_loop_binary_grid(x, y, maxiter, nopt, kernelfun, theta, @random, ninit, theta_lb, theta_ub, lb, ub, seed);
[~,~, cum_regret_BALD(s,:), score_BALD(s,:)]= AL_loop_binary_grid(x, y, maxiter, nopt, kernelfun, theta, @BALD_grid, ninit, theta_lb, theta_ub, lb, ub, seed);
end



i=0;
mr = 2;
mc = 3;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(1)]);
fig.Color =  [1 1 1];
colororder(fig, C)
layout1 = tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');
nexttile(layout1, 1, [1,2]);
i=i+1;
%         errorshaded(x, mu_y, sqrt(sigma2_y), 'Color', cmap(1,:),'DisplayName','Prediction', 'Opacity', 0.2); hold on;
p1 = plot(x,mu_c,'LineWidth',linewidth); hold on;
%         p2 = plot(x,g(x),'LineWidth',linewidth,'Color', cmap(1,:)); hold on;
p2 = plot(x,y,'LineWidth',linewidth); hold on;

scatter(xtrain(ctrain == 1), ctrain(ctrain == 1), markersize, 'MarkerFaceColor', 'k', 'MarkerEdgeColor','k'); hold on;
scatter(xtrain(ctrain == 0), ctrain(ctrain == 0), markersize, 'MarkerFaceColor', 'w', 'MarkerEdgeColor','k') ; hold off;
xticks([0,1])
xticklabels({'0', '1'})
yticks([0,1])
yticklabels({'0', '1'})
grid off
box off
legend([p1, p2],'$\mu_c(x)$', '$P(c=1|x)$', 'Location', 'northeast');
legend boxoff
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
set(gca,'Fontsize',Fontsize)

nexttile(layout1, 4, [1,2]);
i=i+1;

h1 = plot(x,L./max(L),'LineWidth',linewidth); hold on;
h2 = plot(x,var_muc./max(var_muc),'LineWidth',linewidth); hold on;
h3 = plot(x,TME./max(TME),'LineWidth',linewidth); hold on;

[a,b]= max(L./max(L));
vline(x(b),'Linewidth',linewidth, 'ymax', a); hold on;

[a,b]= max(var_muc./max(var_muc));
vline(x(b),'Linewidth',linewidth, 'ymax', a); hold off;
xticks([0,1])
xticklabels({'0', '1'})
% yticks([0,1])
% yticklabels({'0', '1'})
ylabel('Utility function')
legend([h1 h2 h3], '$I(c,f|x, \mathcal{D})$','$V(\mu_c(x)|\mathcal{D})$', 'TME', 'Location', 'northeast');

xlabel('$x$')
box off
legend boxoff
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
set(gca,'Fontsize',Fontsize)

nexttile(layout1, 3, [2,1]);
i=i+1;

options.handle = fig;
options.alpha = 0.2;
options.error= 'sem'; %std
options.line_width = linewidth;
options.color_area = C(1,:);
options.color_line = C(1,:);
h1 = plot_areaerrorbar(cum_regret_maxvar, options); hold on;
options.color_area = C(2,:);
options.color_line = C(2,:);
h2 = plot_areaerrorbar(cum_regret_rand, options); hold on;
options.color_area = C(3,:);
options.color_line = C(3,:);
h3 = plot_areaerrorbar(cum_regret_BALD, options); hold on;
options.color_area = C(5,:);
options.color_line = C(5,:);
h4 = plot_areaerrorbar(cum_regret_TME, options); hold off;

legend([h1 h2 h3 h4], 'Maximum variance', 'Random', 'BALD', 'TME', 'Location', 'northwest');
%legend('EI', 'Random')
xlabel('Iteration','Fontsize',Fontsize)
ylabel('Cumulative regret','Fontsize',Fontsize)
set(gca, 'Fontsize', Fontsize, 'Xlim', [0, maxiter])
grid off
box off
legend boxoff
text(legend_pos(1)-0.1, legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)


fig = figure();
options.handle = fig;
options.alpha = 0.2;
options.error= 'sem'; %std
options.line_width = linewidth;
options.color_area = C(1,:);
options.color_line = C(1,:);
h1 = plot_areaerrorbar(score_maxvar, options); hold on;
options.color_area = C(2,:);
options.color_line = C(2,:);
h2 = plot_areaerrorbar(score_rand, options); hold on;
options.color_area = C(3,:);
options.color_line = C(3,:);
h3 = plot_areaerrorbar(score_BALD, options); hold on;
options.color_area = C(5,:);
options.color_line = C(5,:);
h4 = plot_areaerrorbar(score_TME, options); hold off;

legend([h1 h2 h3 h4], 'Maximum variance', 'Random', 'BALD', 'TME', 'Location', 'northwest');
%legend('EI', 'Random')
xlabel('Iteration','Fontsize',Fontsize)
ylabel('Score','Fontsize',Fontsize)
set(gca, 'Fontsize', Fontsize, 'Xlim', [0, maxiter])
grid off
box off
legend boxoff


% 
% figname  = 'Active_learning';
% folder = [figure_path,figname];
% savefig(fig, [folder,'/', figname, '.fig'])
% exportgraphics(fig, [folder,'/' , figname, '.pdf']);
% exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);

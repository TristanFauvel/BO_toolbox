clear all
add_bo_module
graphics_style_paper
figure_path = '/home/tfauvel/Documents/PhD/Figures/Thesis_figures/Chapter_1/';
close all


letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
legend_pos = [-0.1,1];

%% Define the range of parameters
n = 100;
x = linspace(0,1, n);
d =1;


kernel_name = 'ARD';


ncov_hyp=2;
kernelfun = @ARD_kernelfun;
theta_true = [5,7];%[3,2];

theta.cov = theta_true; %%%%%%%%%%%%%%%%%%%% Known hyperparameters of the covariance

meanfun= @constant_mean;
nmean_hyp = 1;
theta.mean = zeros(nmean_hyp,1);
lb = [-10,0, -5] ; % lengthscale, k0, mu
ub = [10,100, 5] ;

c= othercolor('GnBu7');

rng(5)
regularization = 'nugget';

%generate a function
y = mvnrnd(meanfun(x,0), kernelfun(theta_true, x,x, 'true', regularization));
y = y - mean(y);

x_test = x;
y_test = y;


maxiter=30;
nopt =0; %set nopt to maxiter +1 to get random acquisition
s = rng;

idx= randsample(n,maxiter); % for random sampling


if numel(idx)==1
    i_tr = idx;
else
    i_tr= randsample(idx,1);
end
new_x = x(:,i_tr);
new_y = y(:,i_tr);

x_tr = [];
y_tr = [];
for i =1:2
    x_tr = [x_tr, new_x];
    y_tr = [y_tr, new_y];
    
    
    [mu_y, sigma2_y, ~, ~, Sigma2_y]= prediction(theta, x_tr, y_tr, x, kernelfun, meanfun, [], regularization);
    
    
    if i> nopt
        sigma2_y(sigma2_y<0)=0;
        bestf = max(y_tr);
        EI  = expected_improvement(mu_y, sigma2_y,[], [], bestf, 'max');
        [max_EI, new_i] = max(EI);
        new_x= x(:, new_i);
        new_y = y(x==new_x);
    else
        i_tr= idx(i); %random sampling
        new_x = x(:,i_tr);
        new_y = y(:,i_tr); % no noise %%%%%%%%%%%%%%%%%%%
        
        
    end
end


maxiter = 30;

nreps = 100;
cum_regret_EI= NaN(nreps,maxiter+1);
cum_regret_rand= NaN(nreps,maxiter+1);
cum_regret_TS= NaN(nreps,maxiter+1);

ninit = maxiter+1;
for s = 1:nreps
    rng(s)
    [~,~, cum_regret_EI(s,:)]= BO_loop_grid(n,maxiter, nopt, kernelfun, meanfun, theta, x, y, 'EI', ninit);
    [~,~, cum_regret_rand(s,:)]= BO_loop_grid(n,maxiter, nopt, kernelfun, meanfun,theta,  x, y, 'random', ninit);
    [~,~, cum_regret_TS(s,:)]= BO_loop_grid(n,maxiter, nopt, kernelfun, meanfun,theta,  x, y, 'TS', ninit);
end


rng(2)
sample = mvnrnd(mu_y,Sigma2_y);
[~, new_i_TS] = max(sample);
new_x_TS = x(:, new_i_TS);
max_TS = max(sample);


mr = 2;
mc = 3;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(1)]);
fig.Color =  [1 1 1];
layout1 = tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');
i = 0;

Xlim = [0,1];
nexttile(layout1, 1, [1,2]);
i=i+1;
% errorshaded(x,mu_y, sqrt(sigma2_y), 'Color',  C(1,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold on
plot_gp(x,mu_y, sigma2_y, C(1,:), linewidth); hold on
plot(x, y, 'Color',  C(2,:),'LineWidth', linewidth); hold on;
plot(x_tr, y_tr, 'ro', 'MarkerSize', 10, 'color', C(2,:)); hold on;
scatter(x_tr, y_tr, markersize, C(2,:), 'filled'); hold on;
% xlabel('$x$')
ylabel('$f(x)$')
set(gca, 'Fontsize', Fontsize, 'Xlim', Xlim); %,  'Ylim',Ylim)
grid off
box off
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
set(gca, 'Xtick', [0,1])

nexttile(layout1, 4, [1,2]);
i=i+1;
yyaxis left
p1 = plot(x,sample,'LineWidth',1.5,'Color', C(2,:)); hold on;
vline(new_x_TS,'Linewidth',linewidth, 'ymax', max_TS,  'LineStyle', '--', ...
    'Linewidth', 1); hold off;
xlabel('$x$','Fontsize',Fontsize)
[xt,b] = sort([0,new_x_TS, new_x, 1]);
xticks(xt)
lgs = {'0', '$x_{t+1}$ (TS)', '$x_{t+1}$ (EI)','1'};
xticklabels(lgs(b))
set(gca, 'Fontsize', Fontsize, 'Xlim', Xlim, 'Ylim', [1.1*min(sample), 1.1*max(sample)])
grid off
box off
ax = gca;
ax.YAxis(1).Color = 'k';
ylabel('$\tilde{f}(x)$','Fontsize',Fontsize, 'color', C(2,:))


yyaxis right
p2 = plot(x,EI,'LineWidth',1.5,'Color', C(1,:)); hold on;
vline(new_x,'Linewidth',linewidth, 'ymax', max_EI, 'LineStyle', '--', ...
    'Linewidth', 1); hold off;
% xlabel('$x$','Fontsize',Fontsize)
set(gca, 'Fontsize', Fontsize, 'Xlim', Xlim, 'Ylim', [min(EI)-1, 1+max(EI)])
grid off
box off
% xticks([0,new_x,1])
ax = gca;
ax.YAxis(2).Color = 'k';
ylabel('$EI(x)$','Fontsize',Fontsize, 'color', C(1,:))

% xticklabels({'0', '$x_{t+1}$','1'})
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)


% legend([p1, p2], {'$\tilde{f}(x)$', '$EI(x)$'},'Fontsize',Fontsize)

% legend boxoff

nexttile(layout1, 3, [mr,1]);
i=i+1;

options.handle = fig;
options.alpha = 0.2;
options.error= 'sem'; %std
options.line_width = linewidth;
options.color_area = C(1,:);
options.color_line = C(1,:);
h1 = plot_areaerrorbar(cum_regret_EI, options); hold on;
options.color_area = C(2,:);
options.color_line = C(2,:);
h2 = plot_areaerrorbar(cum_regret_rand, options); hold on;
options.color_area = C(3,:);
options.color_line = C(3,:);
h3 = plot_areaerrorbar(cum_regret_TS, options); hold on;

legend([h1 h2 h3], 'EI', 'Random', 'TS', 'Location', 'northwest');
%legend('EI', 'Random')
xlabel('Iteration','Fontsize',Fontsize)
ylabel('Cumulative regret','Fontsize',Fontsize)
set(gca, 'Fontsize', Fontsize, 'Xlim', [0, maxiter])
grid off
box off
legend boxoff
set(gca, 'Xlim', [1, maxiter])
text(legend_pos(1)-0.2, legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)


figname  = 'Bayesian_optimization';
folder = [figure_path,figname];
savefig(fig, [folder,'/', figname, '.fig'])
exportgraphics(fig, [folder,'/' , figname, '.pdf']);
exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);


clear all
add_gp_module
figure_path = '/home/tfauvel/Documents/PhD/Figures/Thesis_figures/Chapter_1/';
close all
rng(1)
graphics_style_paper;
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

link = @normcdf; %inverse link function
regularization = 'nugget';
post = [];
%% Define the range of parameters
n = 10;
x = linspace(0,1, n);
d =1;

[p,q]= meshgrid(x);
x2d = [p(:), q(:)]';


modeltype = 'exp_prop'; % Approximation method
original_kernelfun =  @Matern52_kernelfun;%kernel used within the preference learning kernel, for subject = computer
x0 = 0;

base_kernelname = 'Matern52';


approximationimation = 'RRGP';

base_kernelfun = @(theta, xi, xj, training, reg) conditioned_kernelfun(theta, original_kernelfun, xi, xj, training, x0, reg);
% kernelfun = @(theta, xi, xj, training) preference_kernelfun(theta, base_kernelfun, xi, xj, training);
kernelfun= @(theta, xi, xj, training, reg) conditional_preference_kernelfun(theta, original_kernelfun, xi, xj, training, reg,x0);

link = @normcdf; %inverse link function for the classification model

% gfunc = @(x) forretal08(x)/10;
% gfunc = @(x) normpdf(x, 0.5, 0.2);
% g = gfunc(x)-gfunc(x0);

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

model.regularization = 'nugget';
model.kernelfun = kernelfun;
model.link = link;
model.modeltype = modeltype;

post = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), [], []);
[mu_c_cond,  mu_f_cond, sigma2_f_cond] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), x2d, post);
[mu_c_cond_x0,  mu_g_cond, sigma2_g_cond, Sigma2_g_cond, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), [x; x0*ones(d,n^d)], post);

%%
init_guess = [];
options.method = 'lbfgs';
options.verbose = 1;
ncandidates = 5;
model.lb_norm = 0;
model.ub_norm = 1;
lb_norm = [model.lb_norm; model.lb_norm];
ub_norm = [model.ub_norm; model.ub_norm];
model.condition.x0 = 0;
model.condition.y0 = 0;
model.type = 'preference';
[xbest, ybest] = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain, ctrain, x, model, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);
ybest = -ybest;

c0 = [ctrain; 0];
c1 = [ctrain;1];
KG= zeros(1, size(x2d,2));

mu_c = model.prediction(theta, xtrain, ctrain, x2d, post);
C= soft_copeland_score(reshape(mu_c, n, n));
C= max(C);

for i = 1:size(x2d,2)
    % KG(i) = knowledge_grad(theta, xtrain, ctrain, x2d(:,i),model, post, c0, c1, xbest, ybest,model.lb_norm, model.ub_norm);
    KG(i) = knowledge_grad_grid(theta, xtrain, ctrain, x2d, x2d(:,i),model, post, c0, c1, ybest);
    
    CEI(i) = copeland_ei(theta, xtrain, ctrain, x2d(:,i), x2d, model, post, C,n);
end


[~,idx_KG] = max(KG);
[~,idx_CEI] = max(CEI);

x_KG = x2d(:, idx_KG);
x_CEI = x2d(:, idx_CEI);

[iKG,jKG] = ind2sub([n, n], idx_KG);
[iCEI,jCEI] = ind2sub([n, n], idx_CEI);


%%
%% Find the true global optimum of g
% [gmax, id_xmax] = max(g);
% xmax = x(id_xmax);
legend_pos = [-0.2,1];
graphics_style_paper;

mr = 2;
mc = 3;
i = 0;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth 1.1*fheight(mr)]);
fig.Color =  [1 1 1];


tiledlayout(mr,mc, 'TileSpacing' , 'tight', 'Padding', 'tight')

cl = [0,1];
nexttile(1)
i=i+1;
imagesc(x, x, reshape(link(f),n,n), cl); hold on;
xlabel('$x$','Fontsize', Fontsize)
ylabel('$x''$','Fontsize', Fontsize)
title('$P(x>x'')$','Fontsize', Fontsize)
set(gca,'YDir','normal')
set(gca,'XTick',[0 0.5 1],'YTick',[0 0.5 1],'Fontsize', Fontsize)
pbaspect([1 1 1])
c = colorbar;
c.Limits = [0,1];
set(c, 'XTick', [0,1]);
colormap(cmap)
p1 = scatter(xtrain(1, ctrain(1:ntr)==1),xtrain(2, ctrain(1:ntr)==1), markersize, 'o', 'k','filled'); hold on;
p2 = scatter(xtrain(1, ctrain(1:ntr)==0),xtrain(2, ctrain(1:ntr)==0), markersize, 'o','k'); hold off;
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
legend([p1, p2], '$c=1$', '$c=0$','Fontsize',Fontsize, 'Location', 'northeast')
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
legend box off


nexttile(2)
i=i+1;
imagesc(x, x, reshape(minmax_normalize(KG),n,n), cl); hold on;
xlabel('$x$','Fontsize', Fontsize)
ylabel('$x''$','Fontsize', Fontsize)
title('$P(x>x'')$','Fontsize', Fontsize)
set(gca,'YDir','normal')
set(gca,'XTick',[0 0.5 1],'YTick',[0 0.5 1],'Fontsize', Fontsize)
pbaspect([1 1 1])
c = colorbar();
% c.Limits = [0,1];
% set(c, 'XTick', [0,1]);
colormap(cmap)
p1 = scatter(xtrain(1, ctrain(1:ntr)==1),xtrain(2, ctrain(1:ntr)==1), markersize, 'o', 'k','filled'); hold on;
p2 = scatter(xtrain(1, ctrain(1:ntr)==0),xtrain(2, ctrain(1:ntr)==0), markersize, 'o','k'); hold off;
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
legend([p1, p2], '$c=1$', '$c=0$','Fontsize',Fontsize, 'Location', 'northeast')
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
legend box off


nexttile(3)
i=i+1;
imagesc(x, x, reshape(minmax_normalize(CEI),n,n), cl); hold on;
xlabel('$x$','Fontsize', Fontsize)
ylabel('$x''$','Fontsize', Fontsize)
title('$P(x>x'')$','Fontsize', Fontsize)
set(gca,'YDir','normal')
set(gca,'XTick',[0 0.5 1],'YTick',[0 0.5 1],'Fontsize', Fontsize)
pbaspect([1 1 1])
c = colorbar();
% c.Limits = [0,1];
% set(c, 'XTick', [0,1]);
colormap(cmap)
p1 = scatter(xtrain(1, ctrain(1:ntr)==1),xtrain(2, ctrain(1:ntr)==1), markersize, 'o', 'k','filled'); hold on;
p2 = scatter(xtrain(1, ctrain(1:ntr)==0),xtrain(2, ctrain(1:ntr)==0), markersize, 'o','k'); hold off;
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
legend([p1, p2], '$c=1$', '$c=0$','Fontsize',Fontsize, 'Location', 'northeast')
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
legend box off


nexttile(5)
i=i+1;
[mu_c_cond_x0,  mu_g_cond, sigma2_g_cond, Sigma2_g_cond, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), [x; x(iKG)*ones(d,n^d)], post);

h1 = plot_gp(x, mu_g_cond, sigma2_g_cond, C(1,:), linewidth); hold on;
h2 = plot(x, g-g(iKG), 'color', C(2,:), 'linewidth', linewidth);  hold on;


xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$g(x)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
%title('Inferred value function $g(x)$','Fontsize', Fontsize)
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
box off
% pbaspect([1 1 1])
vline(x(iKG),'Linewidth',linewidth, 'ymax', mu_g_cond(iKG),  'LineStyle', '--', ...
    'Linewidth', 1); hold on;
vline(x(jKG),'Linewidth',linewidth, 'ymax', mu_g_cond(jKG),  'LineStyle', '--', ...
    'Linewidth', 1); hold on;

[xt,b] = sort([0,x(iKG), x(jKG), 1]);
xticks(unique(xt))
lgs = {'0', '$x_1$', '$x_2$' '1'};

xticklabels(lgs(b))
legend([h1 h2], 'Posterior GP : $p(g|\mathcal{D})$','True value function $g(x)$')
legend box off

nexttile(6)
i=i+1;
[mu_c_cond_x0,  mu_g_cond, sigma2_g_cond, Sigma2_g_cond, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), [x; x(iCEI)*ones(d,n^d)], post);

h1 = plot_gp(x, mu_g_cond, sigma2_g_cond, C(1,:), linewidth); hold on;
h2 = plot(x, g-g(iCEI), 'color', C(2,:), 'linewidth', linewidth);  hold on;


xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$g(x)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
%title('Inferred value function $g(x)$','Fontsize', Fontsize)
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
box off
% pbaspect([1 1 1])
vline(x(iCEI),'Linewidth',linewidth, 'ymax', mu_g_cond(iCEI),  'LineStyle', '--', ...
    'Linewidth', 1); hold on;
vline(x(jCEI),'Linewidth',linewidth, 'ymax', mu_g_cond(jCEI),  'LineStyle', '--', ...
    'Linewidth', 1); hold on;

[xt,b] = sort([0,x(iCEI), x(jCEI), 1]);
xticks(unique(xt))

lgs = {'0', '$x_1$', '$x_2$' '1'};
xticklabels(lgs(b))
legend([h1 h2], 'Posterior GP : $p(g|\mathcal{D})$','True value function $g(x)$')
legend box off

%%



mr = 1;
mc = 3;
i = 0;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth 1.1*fheight(mr)]);
fig.Color =  [1 1 1];


tiledlayout(mr,mc, 'TileSpacing' , 'tight', 'Padding', 'tight')

nexttile
i=i+1;

h1 = plot_gp(x, mu_g_cond, sigma2_g_cond, C(1,:), linewidth); hold on;
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
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
box off
% pbaspect([1 1 1])
vline(max_x,'Linewidth',linewidth, 'ymax', max_y,  'LineStyle', '--', ...
    'Linewidth', 1); hold off;
[xt,b] = sort([0,x1 1]);
xticks(xt)
lgs = {'0', '$x_1$', '1'};
xticklabels(lgs(b))
legend([h1 h2], 'Posterior GP : $p(g|\mathcal{D})$','True value function $g(x)$')
legend box off

[mu_c_cond_x0,  mu_g_cond, sigma2_g_cond, Sigma2_g_cond, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), [x; max_x*ones(d,n^d)], post);

nexttile
legend_pos = [-0.1,1];

i=i+1;
Y = normcdf(mvnrnd(mu_g_cond,Sigma2_g_cond,10000));

[p1, p2] = plot_distro(x, mu_c_cond_x0, Y, C(3,:), C(1,:), linewidth); hold on;
p3 = plot(x, normcdf(g-g(idx)), 'color', C(2,:), 'linewidth', linewidth);  hold off;

xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$P(c=1|x,x_1)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
%title('Inferred value function $g(x)$','Fontsize', Fontsize)
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
box off
% pbaspect([1 1 1])
legend([p3, p2, p1], '$P(x>x_1)$', '$p(\Phi[f(x,x_1)]|\mathcal{D})$', '$\mu_c(x, x_1)$')
legend box off
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
[xt,b] = sort([0,x1, 1]);
xticks(xt)
lgs = {'0', '$x_1$', '1'};
xticklabels(lgs(b))


[max_y,b] = max(var_muc);
max_x = x(b);


legend_pos = [-0.2,1];

nexttile
i=i+1;
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
    'Linewidth', 1); hold off;
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
x2 = max_x;
[xt,b] = sort([0,x1, x2, 1]);
xticks(xt)
lgs = {'0', '$x_1$', '$x_2$','1'};
xticklabels(lgs(b))
% legend([p1, p2], 'Epistemic uncertainty V$[\Phi(f(x,x_1))|\mathcal{D}]$', 'Aleatoric uncertainty V$(c| x, x_1, \mathcal{D})$')
legend([p1, p2], 'V$[\Phi(f(x,x_1))|\mathcal{D}]$', 'V$(c| x, x_1, \mathcal{D})$')

legend box off


figname  = 'Copeland_vs_PKG';
folder = [figure_path,figname];
savefig(fig, [folder,'/', figname, '.fig'])
exportgraphics(fig, [folder,'/' , figname, '.pdf']);
exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);


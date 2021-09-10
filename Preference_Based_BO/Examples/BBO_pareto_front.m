clear all
add_bo_module;
graphics_style_paper;
figure_path = '/home/tfauvel/Documents/PhD/Figures/Thesis_figures/Chapter_1/';

close all
rng(4)

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

link = @normcdf; %inverse link function

%% Define the range of parameters
n = 2000;
x = linspace(0,1, n);
d =1;
ntr = 15;

x0 = x(:,1);

modeltype = 'exp_prop'; % Approximation method
kernelfun =  @Matern52_kernelfun;%kernel used within the preference learning kernel, for subject = computer
kernelname = 'Matern52';

link = @normcdf; %inverse link function for the classification model
model.regularization = 'nugget';
model.kernelfun = kernelfun;

model.link = link;
model.modeltype = modeltype;
model.kernelname = kernelname;
model.lb_norm  = 0;
model.ub_norm  = 1;
model.lb = 0;
model.ub = 1;
model.ns = 0;
model.task = 'max';
post = [];
model.D = 1;
regularization = 'none';
theta= [-1;1];
g = mvnrnd(zeros(1,n),kernelfun(theta, x, x, 'false', regularization));

if strcmp(model.kernelname, 'Matern52') || strcmp(model.kernelname, 'Matern32') || strcmp(model.kernelname, 'ARD')
    approximation.method = 'RRGP';
else
    approximation.method = 'SSGP';
end
approximation.decoupled_bases = 1;
approximation.nfeatures = 256;
[approximation.phi, approximation.dphi_dx] = sample_features_GP(theta(:), model, approximation);


figure()
subplot(1,2,1)
plot(g);
subplot(1,2,2)
plot(normcdf(g))

nsamp= 1000;
rd_idx = randsample(n, nsamp, 'true');
xtrain= x(:,rd_idx);
ytrain= g(rd_idx)';
ctrain = link(ytrain)>rand(nsamp,1);


[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), x, model, post);

sample_g = sample_GP(theta, x, g(:), model, approximation);

xx = mu_y;
yy = sqrt(var_muc);

figure()
scatter(xx, yy)

%% Find the true global optimum of g
[gmax, id_xmax] = max(g);
xmax = x(id_xmax);

% GP = [mu_y, sqrt(sigma2_y)]';
GP = [xx, yy]';
not_pareto = [];
idx_pareto = [];
pareto_front = [];
for i = 1:n
    if ~any(sum(GP(:,i)< GP(:,setdiff(1:n,i)),1)==2)
        pareto_front = [pareto_front, GP(:,i)];
        idx_pareto = [idx_pareto,i];
    else
        not_pareto  = [not_pareto, GP(:,i)];
    end
end


legend_pos = [-0.2,1];
Y = normcdf(mvnrnd(mu_y,Sigma2_y,5000));

[new_EI, EI] = EI_Tesch(theta, xtrain, ctrain,model, post, approximation);
[new_TS, TS] = TS_binary(theta, xtrain, ctrain,model, post, approximation);
[new_UCB, UCB] = UCB_binary(theta, xtrain, ctrain,model, post, approximation);
[new_bivariateEI, bivariateEI] = bivariate_EI_binary(theta, xtrain, ctrain,model, post);

mr = 1;
mc = 2;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(1)]);
fig.Color =  [1 1 1];
layout1 = tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');
i = 0;
Xlim = [0,1];

nexttile();
i=i+1;
[p1, p2] = plot_distro(x, mu_c, Y, C(3,:), C(1,:), linewidth); hold on;
% p3 = plot(x, normcdf(g), 'color', C(2,:), 'linewidth', linewidth);  hold on;
p3 = scatter(x(idx_pareto), normcdf(g(idx_pareto)), markersize, C(2,:),'filled');  hold on;
p4 = scatter(x(setdiff(1:n,idx_pareto)), normcdf(g(setdiff(1:n,idx_pareto))), markersize, 'k','filled');  hold on;
h2 = scatter(new_EI, normcdf(sample_g(new_EI)), 10*markersize, 'b', '+'); hold on;
h3 = scatter(new_TS, normcdf(sample_g(new_TS)), 10*markersize, 'y','x'); hold on;
h4 = scatter(new_UCB, normcdf(sample_g(new_UCB)), 10*markersize,'g', 'o'); hold on;
% h5 = scatter(new_bivariateEI, normcdf(sample_g(new_bivariateEI)), 10*markersize, 'd'); hold on;


xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$P(c=1|x,x_1)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
%title('Inferred value function $g(x)$','Fontsize', Fontsize)
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
box off
% pbaspect([1 1 1])
legend([p3, p2, p1], '$P(x>x_1)$, Pareto front', '$P(x>x_1|\mathcal{D})$', '$\mu_c(x, x_1)$')
legend box off
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)


nexttile();
s = scatter(not_pareto(1,:),not_pareto(2,:), markersize, 'k', 'filled'); hold on;
% h1 = plot(pareto_front(1,:),pareto_front(2,:),'color', C(1,:), 'linewidth', linewidth); hold on;
h1 = scatter(pareto_front(1,:),pareto_front(2,:),markersize, C(2,:), 'filled'); hold on;
box off
xlabel('$\mu_y$')
ylabel('V$[\Phi(f(x))|\mathcal{D}]$')
set(gca, 'Fontsize', Fontsize);

%%
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_EI, model, post);
h2 = scatter(mu_y, sqrt(var_muc), 10*markersize,'b', '+'); hold on;
%%
rng(1)
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_TS, model, post);
h3 = scatter(mu_y, sqrt(var_muc), 10*markersize, 'y','x'); hold on;

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_UCB, model, post);
h4 = scatter(mu_y, sqrt(var_muc), 10*markersize,'g', 'o'); hold on;

% [mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_bivariateEI, model, post);
% h5 = scatter(mu_y, sqrt(var_muc), 10*markersize, 'd'); hold on;

legend([h1 h2 h3 h4], 'Pareto front', 'EI (Tesch 2013)', 'Thompson sampling', 'UCB$_{\Phi}$');%, 'Bivariate EI')
legend box off
s.AlphaData = 0.5;
s.MarkerFaceAlpha = 'flat';

% figname  = 'Pareto_front';
% folder = [figure_path,figname];
% savefig(fig, [folder,'/', figname, '.fig'])
% exportgraphics(fig, [folder,'/' , figname, '.pdf']);
% exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);


%%

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), x, model, post);

xx = mu_c;
yy = sqrt(var_muc);

GP = [xx, yy]';

pareto_front = [];
not_pareto = [];
for i = 1:n
    if ~any(sum(GP(:,i)< GP(:,setdiff(1:n,i)),1)==2)
        pareto_front = [pareto_front, GP(:,i)];
    else
        not_pareto  = [not_pareto, GP(:,i)];
    end
end

mr = 1;
mc = 2;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(1)]);
fig.Color =  [1 1 1];
layout1 = tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');
i = 0;
Xlim = [0,1];

nexttile();
i=i+1;
[p1, p2] = plot_distro(x, mu_c, Y, C(3,:), C(1,:), linewidth); hold on;
% p3 = plot(x, normcdf(g), 'color', C(2,:), 'linewidth', linewidth);  hold on;

p3 = scatter(x(idx_pareto), normcdf(g(idx_pareto)), markersize, C(2,:),'filled');  hold on;
p4 = scatter(x(setdiff(1:n,idx_pareto)), normcdf(g(setdiff(1:n,idx_pareto))), markersize, 'k','filled');  hold on;

h2 = scatter(new_EI, normcdf(sample_g(new_EI)), 10*markersize, 'b', '+'); hold on;
h3 = scatter(new_TS, normcdf(sample_g(new_TS)), 10*markersize, 'y','x'); hold on;
h4 = scatter(new_UCB, normcdf(sample_g(new_UCB)), 10*markersize, 'g','o'); hold on;
% h5 = scatter(new_bivariateEI, normcdf(sample_g(new_bivariateEI)), 10*markersize, 'd'); hold on;


xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$P(c=1|x,x_1)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
%title('Inferred value function $g(x)$','Fontsize', Fontsize)
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
box off
% pbaspect([1 1 1])
legend([p3, p2, p1], '$P(x>x_1)$', '$P(x>x_1|\mathcal{D})$', '$\mu_c(x, x_1)$')
legend box off
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)


nexttile();
s = scatter(not_pareto(1,:),not_pareto(2,:), markersize, 'k', 'filled'); hold on;
% h1 = plot(pareto_front(1,:),pareto_front(2,:),'color', C(1,:), 'linewidth', linewidth); hold on;
h1 = scatter(pareto_front(1,:),pareto_front(2,:),markersize, C(2,:), 'filled'); hold on;
box off
xlabel('E$[\Phi(f(x))|\mathcal{D}]$')
ylabel('V$[\Phi(f(x))|\mathcal{D}]$')
set(gca, 'Fontsize', Fontsize);

%%
 
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_EI, model, post);
h2 = scatter(mu_c, sqrt(var_muc), 10*markersize,'b', '+'); hold on;

rng(1)
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_TS, model, post);
h3 = scatter(mu_c, sqrt(var_muc), 10*markersize, 'y','x'); hold on;

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_UCB, model, post);
h4 = scatter(mu_c, sqrt(var_muc), 10*markersize,'g', 'o'); hold on;

% [mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_bivariateEI, model, post);
% h5 = scatter(mu_c, sqrt(var_muc), 10*markersize, 'd'); hold on;
% 

s.AlphaData = 0.5;
s.MarkerFaceAlpha = 'flat';

legend([h1 h2 h3 h4], 'Pareto front', 'EI (Tesch 2013)', 'Thompson sampling', 'UCB$_{\Phi}$') %, 'Bivariate EI')
legend box off

figname  = 'Pareto_front';
folder = [figure_path,figname];
savefig(fig, [folder,'/', figname, '.fig'])
exportgraphics(fig, [folder,'/' , figname, '.pdf']);
exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);


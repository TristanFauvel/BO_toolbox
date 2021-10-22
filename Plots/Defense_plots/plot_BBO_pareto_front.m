clear all
add_bo_module;
graphics_style_presentation;
 
close all
rng(4)

currentFile = mfilename( 'fullpath' );
[pathname,~,~] = fileparts( currentFile );
 figure_path = [pathname, '/Figures/'];
savefigures = 1;

link = @normcdf; %inverse link function

n = 3000;
ub  = 1;
lb = 0;
x = linspace(lb, ub, n);
d =1;
ntr = 30;

x0 = x(:,1);

modeltype = 'exp_prop'; % Approximation method
kernelfun =  @Matern52_kernelfun;%kernel used within the preference learning kernel, for subject = computer
kernelname = 'Matern52';

link = @normcdf; %inverse link function for the classification model
regularization = 'nugget';
meanfun = 0;
type = 'classification';
hyps.ncov_hyp =2; % number of hyperparameters for the covariance function
hyps.nmean_hyp =0; % number of hyperparameters for the mean function
hyps.hyp_lb = -10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
hyps.hyp_ub = 10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
D = 1;
% model.task = 'max';
post = [];
model.D = 1;
regularization = 'none';
theta.cov= [-1;1];

model = gp_classification_model(D, meanfun, kernelfun, regularization, hyps, lb, ub, type, link, modeltype, kernelname, []);


g = -0.5*mvnrnd(zeros(1,n),kernelfun(theta.cov, x, x, 'false', regularization));

if strcmp(model.kernelname, 'Matern52') || strcmp(model.kernelname, 'Matern32') || strcmp(model.kernelname, 'ARD')
    approximation.method = 'RRGP';
else
    approximation.method = 'SSGP';
end
approximation.decoupled_bases = 1;
approximation.nfeatures = 256;
[approximation.phi, approximation.dphi_dx] = sample_features_GP(theta, model, approximation);


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

xtrain = xtrain(:,1:ntr);
ctrain = ctrain(1:ntr);
ytrain = ytrain(1:ntr);

post = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), [], post);

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), x, post);

sample_g = sample_GP(theta, x, g(:), model, approximation);
legend_pos = [-0.15,1];
Y = normcdf(mvnrnd(mu_y,Sigma2_y,5000));
[new_EI, EI] = EI_Tesch(theta, xtrain, ctrain,model, post, approximation);
[new_TS, TS] = TS_binary(theta, xtrain, ctrain,model, post, approximation);
[new_UCB, UCB] = UCB_binary(theta, xtrain, ctrain,model, post, approximation);
[new_bivariateEI, bivariateEI] = bivariate_EI_binary(theta, xtrain, ctrain,model, post);
[new_UCBf, UCBf] = UCB_binary_latent(theta, xtrain, ctrain,model, post, approximation);
model2 = model;
model2.modeltype = 'laplace';
[new_BKG, bkg] = BKG(theta, xtrain, ctrain,model2, [], approximation);

% xx = mu_y;
% yy = sqrt(var_muc);
% 
% figure()
% scatter(xx, yy)
% 
% %% Find the true global optimum of g
% [gmax, id_xmax] = max(g);
% xmax = x(id_xmax);
% 
% % GP = [mu_y, sqrt(sigma2_y)]';
% GP = [xx, yy]';
% not_pareto = [];
% idx_pareto = [];
% pareto_front = [];
% for i = 1:n
%     if ~any(sum(GP(:,i)< GP(:,setdiff(1:n,i)),1)==2)
%         pareto_front = [pareto_front, GP(:,i)];
%         idx_pareto = [idx_pareto,i];
%          not_pareto  = [not_pareto, NaN*ones(2,1)];
%     else
%         pareto_front = [pareto_front, NaN*ones(2,1)];
%         not_pareto  = [not_pareto, GP(:,i)];
%     end
% end
% 
% 
% 
 % 
% mr = 1;
% mc = 2;
% fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(1)]);
% fig.Color =  background_color;
% layout1 = tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');
% i = 0;
% Xlim = [0,1];
% 
% nexttile();
% i=i+1;
% [p1, p2] = plot_distro(x, mu_c, Y, C(3,:), C(1,:), linewidth); hold on;
% 
% xp = NaN(1,n); 
% yp = NaN(1,n);
% xp(idx_pareto) = x(idx_pareto);
% yp(idx_pareto) = normcdf(g(idx_pareto));
% p3 = plot(xp, yp, 'color', C(2,:), 'linewidth', linewidth);  hold on;
% xp = NaN(1,n); 
% yp = NaN(1,n);
% xp(setdiff(1:n,idx_pareto)) = x(setdiff(1:n,idx_pareto));
% yp(setdiff(1:n,idx_pareto)) = normcdf(g(setdiff(1:n,idx_pareto)));
% p4 = plot(xp, yp, 'color',  0.5*[1 1 1], 'linewidth', linewidth);  hold on;
% 
% h2 = scatter(new_EI, normcdf(sample_g(new_EI)), 10*markersize, 'b', 'x'); hold on;
% h3 = scatter(new_TS, normcdf(sample_g(new_TS)), 10*markersize, 'm','x'); hold on;
% h4 = scatter(new_UCB, normcdf(sample_g(new_UCB)), 10*markersize,'g', 'x'); hold on;
% % h5 = scatter(new_bivariateEI, normcdf(sample_g(new_bivariateEI)), 10*markersize, 'x'); hold on;
% h5 = scatter(new_UCBf, normcdf(sample_g(new_UCBf)), 10*markersize, 'c', 'x'); hold on;
% 
% 
% xlabel('$x$', 'Fontsize', Fontsize)
% ylabel('$P(c=1|x)$', 'Fontsize', Fontsize)
% set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
% ytick = get(gca,'YTick');
% set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
% %title('Inferred value function $g(x)$','Fontsize', Fontsize)
% text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
% box off
% % pbaspect([1 1 1])
% legend([p3, p2, p1], '$P(c= 1|x)$, Pareto front', '$P(c = 1| x, \mathcal{D})$', '$\mu_c(x)$')
% legend box off
% text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
% 
% 
% nexttile();
% % s = scatter(not_pareto(1,:),not_pareto(2,:), markersize, k, 'filled'); hold on;
% h1 = plot(pareto_front(1,:),pareto_front(2,:),'color', C(2,:), 'linewidth', linewidth); hold on;
% h2 = plot(not_pareto(1,:),not_pareto(2,:),'color', 0.5*[1 1 1], 'linewidth', linewidth); hold on;
% 
% box off
% xlabel('$\mu_y$')
% ylabel('V$[\Phi(f(x))|\mathcal{D}]$')
% set(gca, 'Fontsize', Fontsize);
% 
% %%
% [mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_EI, post);
% h2 = scatter(mu_y, sqrt(var_muc), 10*markersize,'b', 'x'); hold on;
% %%
% rng(1)
% [mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_TS, post);
% h3 = scatter(mu_y, sqrt(var_muc), 10*markersize, 'm','x'); hold on;
% 
% [mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_UCB, post);
% h4 = scatter(mu_y, sqrt(var_muc), 10*markersize,'g', 'x'); hold on;
% 
% % [mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_bivariateEI, post);
% % h5 = scatter(mu_y, sqrt(var_muc), 10*markersize, 'x'); hold on;
% [mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_UCBf, post);
% h5 = scatter(mu_y, sqrt(var_muc), 10*markersize,'c', 'x'); hold on;
% 
% legend([h1 h2 h3 h4 h5], 'Pareto front', 'EI (Tesch et al, 2013)', 'Thompson sampling', 'UCB$_{\Phi}$','UCB$_f$');%, 'Bivariate EI')
% legend box off
% s.AlphaData = 0.5;
% s.MarkerFaceAlpha = 'flat';
% 
% % figname  = 'Pareto_front';
% % folder = [figure_path,figname];
% % savefig(fig, [folder,'/', figname, '.fig'])
% % exportgraphics(fig, [folder,'/' , figname, '.pdf']);
% % exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);


%%

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), x, post);

xx = mu_c;
yy = sqrt(var_muc);
not_pareto = [];
idx_pareto = [];

GP = [xx, yy]';

pareto_front = [];
not_pareto = [];
for i = 1:n
    if ~any(sum(GP(:,i)< GP(:,setdiff(1:n,i)),1)==2)
        pareto_front = [pareto_front, GP(:,i)];
        idx_pareto = [idx_pareto,i];
         not_pareto  = [not_pareto, NaN*ones(2,1)];
    else
        pareto_front = [pareto_front, NaN*ones(2,1)];
        not_pareto  = [not_pareto, GP(:,i)];
    end
end


mr = 1;
mc = 2;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(1)]);
fig.Color =  background_color;
layout1 = tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');
i = 0;
Xlim = [0,1];

nexttile();
i=i+1;
[p1, p2] = plot_distro(x, mu_c, Y, C(3,:), C(1,:), linewidth); hold on;

xp = NaN(1,n); 
yp = NaN(1,n);
xp(idx_pareto) = x(idx_pareto);
yp(idx_pareto) = normcdf(g(idx_pareto));
p3 = plot(xp, yp, 'color', C(2,:), 'linewidth', linewidth);  hold on;
xp = NaN(1,n); 
yp = NaN(1,n);
xp(setdiff(1:n,idx_pareto)) = x(setdiff(1:n,idx_pareto));
yp(setdiff(1:n,idx_pareto)) = normcdf(g(setdiff(1:n,idx_pareto)));
p4 = plot(xp, yp, 'color',  0.5*[1 1 1], 'linewidth', linewidth);  hold on;


h2 = scatter(new_EI, normcdf(sample_g(new_EI)), 10*markersize, 'b', 'x','LineWidth',1.5); hold on;
h3 = scatter(new_TS, normcdf(sample_g(new_TS)), 10*markersize, 'm','x','LineWidth',1.5); hold on;
h4 = scatter(new_UCB, normcdf(sample_g(new_UCB)), 10*markersize, 'g','x','LineWidth',1.5); hold on;
h5 = scatter(new_UCBf, normcdf(sample_g(new_UCBf)), 10*markersize, 'c','x','LineWidth',1.5); hold on;
h6 = scatter(new_BKG, normcdf(sample_g(new_BKG)), 10*markersize, 'b','o','LineWidth',1.5); hold on;


xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$P(c=1|x)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
box off
legend([p3, p2, p1], '$P(c=1|x)$, Pareto front', '$p(\Phi(f(x))|\mathcal{D})$', '$\mu_c(x)$')
legend box off


nexttile();
i=i+1;
[ordered_pareto_front(1,:),b] = sort(pareto_front(1,:));
ordered_pareto_front(2,:) = pareto_front(2,b);
plot(ordered_pareto_front(1,:), ordered_pareto_front(2,:), ':', 'linewidth', linewidth, 'color', C(2,:));hold on;

h1 = plot(pareto_front(1,:),pareto_front(2,:),'color', C(2,:), 'linewidth', linewidth); hold on;
h2 = plot(not_pareto(1,:),not_pareto(2,:),'color', 0.5*[1 1 1], 'linewidth', linewidth); hold on;

box off
xlabel('E$[\Phi(f(x))|\mathcal{D}]$')
ylabel('$\sqrt{V[\Phi(f(x))|\mathcal{D}]}$')
set(gca, 'Fontsize', Fontsize);

%%
 
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_EI, post);
h2 = scatter(mu_c, sqrt(var_muc), 10*markersize,'b', 'x','LineWidth',1.5); hold on;

rng(1)
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_TS, post);
h3 = scatter(mu_c, sqrt(var_muc), 10*markersize, 'm','x','LineWidth',1.5); hold on;

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_UCB, post);
h4 = scatter(mu_c, sqrt(var_muc), 10*markersize,'g', 'x','LineWidth',1.5); hold on;

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_UCBf, post);
h5 = scatter(mu_c, sqrt(var_muc), 10*markersize,'c', 'x','LineWidth',1.5); hold on;

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_BKG, post);
h6 = scatter(mu_c, sqrt(var_muc), 10*markersize,'b', 'o','LineWidth',1.5); hold on;

s.AlphaData = 0.5;
s.MarkerFaceAlpha = 'flat';

legend([h1 h2 h3 h4 h5 h6], 'Pareto front', 'EI (Tesch et al, 2013)', 'Thompson sampling', 'UCB$_{\Phi}$', 'UCB$_{f}$ (Tesch et al, 2013)', 'BKG') %, 'Bivariate EI')
legend box off
% 
% figname  = 'Pareto_front';
% folder = [figure_path,figname];
% savefig(fig, [folder,'/', figname, '.fig'])
% exportgraphics(fig, [folder,'/' , figname, '.pdf']);
% exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);

%%

n = 10000;
e = linspace(0,20,n);
for i = 1:n
[new_UCB(i), UCB] = UCB_binary(theta, xtrain, ctrain,model, post, approximation, 'e', e(i));
[new_UCBf(i), UCBf] = UCB_binary_latent(theta, xtrain, ctrain,model, post, approximation, 'e', e(i));
end

% h4 = scatter(new_UCB, normcdf(sample_g(new_UCB)), 10*markersize, 'g','x','LineWidth',1.5); hold on;
% h5 = scatter(new_UCBf, normcdf(sample_g(new_UCBf)), 10*markersize, 'c','x','LineWidth',1.5); hold on;

figure()
plot(ordered_pareto_front(1,:), ordered_pareto_front(2,:), ':', 'linewidth', linewidth, 'color', C(2,:));hold on;

h1 = plot(pareto_front(1,:),pareto_front(2,:),'color', C(2,:), 'linewidth', linewidth); hold on;
h2 = plot(not_pareto(1,:),not_pareto(2,:),'color', 0.5*[1 1 1], 'linewidth', linewidth); hold on;

box off
xlabel('E$[\Phi(f(x))|\mathcal{D}]$')
ylabel('$\sqrt{V[\Phi(f(x))|\mathcal{D}]}$')
set(gca, 'Fontsize', Fontsize);
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_UCB, post);
h4 = scatter(mu_c, sqrt(var_muc), 10*markersize,'g', 'x','LineWidth',1.5); hold on;
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), new_UCBf, post);
h5 = scatter(mu_c, sqrt(var_muc), 10*markersize,'c', 'x','LineWidth',1.5); hold on;
legend([h4, h5], 'UCB$_\Phi$', 'UCB$_f$')

darkBackground(fig,background,[1 1 1])
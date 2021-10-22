clear all
add_bo_module;
graphics_style_presentation;
 
close all
rng(1)

currentFile = mfilename( 'fullpath' );
[pathname,~,~] = fileparts( currentFile );
 figure_path = [pathname, '/Figures/'];
savefigures = 1;

link = @normcdf; %inverse link function

n = 1000;
x = linspace(0,1, n);
d =1;
ntr = 100;

x0 = x(:,1);

modeltype = 'exp_prop'; % Approximation method
kernelfun =  @Matern52_kernelfun;%kernel used within the preference learning kernel, for subject = computer
kernelname = 'Matern52';

link = @normcdf; %inverse link function for the classification model
  


post = [];
regularization = 'none';
theta.cov= [-1;1];

sample_prior = mvnrnd(mu_y_ard, Kard);
meanfun = @contant_mean;
regularization = 'nugget';
type = 'regression';
hyps.ncov_hyp =2; % number of hyperparameters for the covariance function
hyps.nmean_hyp =1; % number of hyperparameters for the mean function
hyps.hyp_lb = -10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
hyps.hyp_ub = 10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
model = gp_regression_model(D, meanfun, kernelfun, regularization, hyps, lb, ub, kernelname);



g = 4*mvnrnd(zeros(1,n),kernelfun(theta.cov, x, x, 'false', regularization));

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

rd_idx = randsample(n, ntr-80, 'true');
rd_idx = [rd_idx; randsample(850:980, 80, 'true')'];
    
xtrain = x(:,rd_idx);
 ytrain = g(rd_idx);
 ctrain = link(ytrain)>rand(1,ntr);

type = 'regression';
hyps.ncov_hyp =2; % number of hyperparameters for the covariance function
hyps.nmean_hyp =1; % number of hyperparameters for the mean function
hyps.hyp_lb = -10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
hyps.hyp_ub = 10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
model = gp_regression_model(D, meanfun, kernelfun, regularization, hyps, lb, ub, kernelname);

 
post = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), [], post);

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, ...
    var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), x, post);

sample_g = sample_GP(theta, x, g(:), model, approximation);
legend_pos = [-0.15,1];
Y = normcdf(mvnrnd(mu_y,Sigma2_y,5000));
[new_EI, EI] = EI_Tesch(theta, xtrain, ctrain,model, post, approximation);
[new_TS, TS] = TS_binary(theta, xtrain, ctrain,model, post, approximation);
[new_UCB, UCB] = UCB_binary(theta, xtrain, ctrain,model, post, approximation);
[new_bivariateEI, bivariateEI] = bivariate_EI_binary(theta, xtrain, ctrain,model, post);
[new_UCBf, UCBf] = UCB_binary_latent(theta, xtrain, ctrain,model, post, approximation);


[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain, ctrain, x, post);



mr = 2;
mc = 2;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  background_color;
layout1 = tiledlayout(mr,mc, 'TileSpacing', 'compact', 'padding','compact');
i = 0;
Xlim = [0,1];

nexttile();
i=i+1;
[p1, p2] = plot_distro(x, mu_c, Y, C(3,:), C(1,:), linewidth); hold on;
p3 =plot(x, normcdf(g), 'color', C(2,:), 'linewidth', linewidth)
% h2 = scatter(new_EI, normcdf(sample_g(new_EI)), 10*markersize, 'b', 'x','LineWidth',1.5); hold on;
% h3 = scatter(new_TS, normcdf(sample_g(new_TS)), 10*markersize, 'm','x','LineWidth',1.5); hold on;
h4 = scatter(new_UCB, normcdf(sample_g(new_UCB)), 10*markersize, 'b','x','LineWidth',1.5); hold on;
h5 = scatter(new_UCBf, normcdf(sample_g(new_UCBf)), 10*markersize, k,'x','LineWidth',1.5); hold on;


xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$P(c=1|x)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
box off
legend([p3, p2, p1, h4, h5], '$P(c=1|x)$', '$p(\Phi(f(x))|\mathcal{D})$', '$\mu_c(x)$' ,'UCB$_\Phi$','UCB$_f$')
legend box off


nexttile();
i=i+1;
[p1, p2] = plot_gp(x, mu_y, sigma2_y, C(1,:),linewidth, 'background', background); hold on;
p3 = plot(x, g, 'color', C(2,:), 'linewidth', linewidth);
h4 = scatter(new_UCB, sample_g(new_UCB), 10*markersize, 'b','x','LineWidth',1.5); hold on;
h5 = scatter(new_UCBf, sample_g(new_UCBf), 10*markersize, k,'x','LineWidth',1.5); hold on;
box off
xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$f(x)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
legend([p3, p1], '$f(x)$', '$p(f(x)|\mathcal{D})$')
legend box off


nexttile();
e = norminv(0.99);
sigma_c = sqrt(var_muc);
ucb_val = mu_c + e*sigma_c;
h2 = plot(x, ucb_val, 'Color',  C(1,:),'LineWidth', linewidth); hold on;
 ylabel('$\alpha(x)$')
box off
xlabel('$x$')
 xlabel('$x$', 'Fontsize', Fontsize)
ylabel('UCB$_\Phi(x)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)

[max_y, b] = max(ucb_val);
max_x = x(b);
vline(max_x,'Linewidth',linewidth, 'ymax', max_y,  'LineStyle', '--', ...
    'Linewidth', 1, 'Color', k); hold off;


nexttile();
sigma_y = sqrt(sigma2_y);
e = 1; % 1 is used in the original paper by Tesch et al (2013). norminv(0.975);
e = norminv(0.99);
ucb_val = mu_y + e*sigma_y;

h2 = plot(x, ucb_val, 'Color',  C(1,:),'LineWidth', linewidth); hold on;
 ylabel('$\alpha$')
box off
xlabel('$x$')
 xlabel('$x$', 'Fontsize', Fontsize)
ylabel('UCB$_f(x)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
[max_y, b] = max(ucb_val);
max_x = x(b);
vline(max_x,'Linewidth',linewidth, 'ymax', max_y,  'LineStyle', '--', ...
    'Linewidth', 1, 'Color', k); hold off;

 
darkBackground(fig,background,[1 1 1])

 figname  = 'GP_regression_3';
 export_fig(fig, [figure_path,'/' , figname, '.pdf']);
export_fig(fig, [figure_path,'/' , figname, '.png']);


%%
N = 75;
sigma2_y_range = linspace(0.001,5,N);
mu_y_range = linspace(-5,5,N);
h = @(p) -p.*log(p+eps) - (1-p).*log(1-p+eps);

[p,q]= meshgrid(mu_y_range, sigma2_y_range);

inputs  = [p(:),q(:)]';
sigma2_y = inputs(2,:); 
mu_y = inputs(1,:); 

mu_c = normcdf(mu_y./sqrt(1+sigma2_y));

h = mu_y./sqrt(1+sigma2_y);
a = 1./sqrt(1+2*sigma2_y);

[tfn_output, dTdh, dTda] = tfn(h, a);
var_muc = (mu_c - 2*tfn_output) - mu_c.^2;
var_muc(var_muc<0 ) = 0;
sigma_y = sqrt(sigma2_y);
e = 1; % 1 is used in the original paper by Tesch et al (2013). norminv(0.975);
e = norminv(0.99);
ucb_f = mu_y + e*sigma_y;
 
e = norminv(0.99);
sigma_c = sqrt(var_muc);
ucb_Phi = mu_c + e*sigma_c;

mr = 1;
mc = 4;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  background_color;
tiledlayout(1,mc, 'TileSpacing', 'compact', 'padding','compact');
nexttile(1);
y_best = 0;
sigma_y = sqrt(sigma2_y);
d = (mu_y - y_best)./sigma_y;
d(sigma_y==0)=0;

normpdf_d =  normpdf(d);
normcdf_d= normcdf(d);

EI = (mu_y - y_best).*normcdf_d+ sigma_y.*normpdf_d;%Brochu
% EI(sigma_y==0) = 0;
imagesc(mu_y_range, sigma2_y_range, reshape(minmax_normalize(EI),N,N)); hold on;
xlabel('$\mu_f(x)$','Fontsize',Fontsize)
ylabel('$\sigma^2_f(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal')
pbaspect([1 1 1])
% cb = colorbar;
title('EI$_f$')
set(gca, 'fontsize', Fontsize)
 
mu_c_best = 0.5;
ybest = norminv(mu_c_best);
nsamps= 1e5;
sigma_y = sqrt(sigma2_y);
ei = zeros(size(sigma_y));

for i = 1:size(ei,2)
    %     samples = mu_y(i) + sigma_y(i).*randn(nsamps,1);
    %     samples(samples<ybest) = [];
    pd = makedist('Normal');
    pd.mu = mu_y(i);
    pd.sigma = sigma_y(i);
    try
    t = truncate(pd,ybest,inf);
    samples= random(t,nsamps,1);
    arg = model.link(samples) - mu_c_best;
    ei(i) = mean(arg);
    catch
        ei(i) = 0;
    end

end

nexttile(2);
imagesc(mu_y_range, sigma2_y_range, reshape(minmax_normalize(ei),N,N)); hold on;
xlabel('$\mu_f(x)$','Fontsize',Fontsize)
% ylabel('$\sigma^2_f(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal')
pbaspect([1 1 1])
% cb = colorbar;
title('EI$_\Phi$')
set(gca, 'fontsize', Fontsize)
set(gca,'ytick',[])


nexttile(3);
imagesc(mu_y_range, sigma2_y_range, reshape(minmax_normalize(ucb_Phi),N,N)); hold on;
xlabel('$\mu_f(x)$','Fontsize',Fontsize)
% ylabel('$\sigma^2_f(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal')
pbaspect([1 1 1])
 cb = colorbar;
cb.FontName = 'CMU Serif';
cb.FontSize = Fontsize;
 
title('UCB$_\Phi$')
set(gca, 'fontsize', Fontsize)
set(gca,'ytick',[])


nexttile(4);
imagesc(mu_y_range, sigma2_y_range, reshape(minmax_normalize(ucb_f),N,N)); hold on;
xlabel('$\mu_f(x)$','Fontsize',Fontsize)
% ylabel('$\sigma^2_f(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal')
pbaspect([1 1 1])
% cb = colorbar;
title('UCB$_f$')
set(gca, 'fontsize', Fontsize)
% set(get(cb,'Title'),'String','(nats)', 'Interpreter', 'latex')
set(gca,'ytick',[])
colormap(cmap)

darkBackground(fig,background,[1 1 1])

figname  = 'BBO_acq_funs_comparison';
  export_fig(fig, [figure_path,'/' , figname, '.pdf']);
export_fig(fig, [figure_path,'/' , figname, '.png']);

 
% set(get(cb,'Title'),'String','(nats)', 'Interpreter', 'latex')

% nexttile();
% imagesc(mu_y_range, sigma2_y_range, reshape(var_muc,N,N)); hold on;
% xlabel('$\mu_f(x)$','Fontsize',Fontsize)
% ylabel('$\sigma^2_f(x)$','Fontsize',Fontsize)
% set(gca,'YDir','normal')
% pbaspect([1 1 1])
% cb = colorbar;
% colormap(cmap)
% title('$V(\Phi(f(x)))$')
% set(gca, 'fontsize', Fontsize)
% 

 
 
%% Sample GP 

close all
add_gp_module;
figure_path = '/home/tfauvel/Documents/PhD/Figures/Thesis_figures/Chapter_1/';

rng(13);%12 
n=200; %resolution
x = linspace(0,1,n);

kernelfun = @Matern52_kernelfun;
kernelname = 'Matern52';
theta.cov = [log(0.8/10),-0.5];
theta.mean= 0;


kernelfun = @ARD_kernelfun;
kernelname = 'ARD';
theta.cov = [-2*log(0.1),0];
lengthscale = exp(-theta.cov(1)/2);


Sigma =kernelfun(theta.cov,x,x, true, 'nugget');

g =  mvnrnd(constant_mean(x,0), kernelfun(theta.cov, x,x, true, 'nugget')); %generate a function

sigma = 0;
y = g + sigma*randn(1,n); %add measurement noise

graphics_style_paper;

 
% h=figure(1);
% h.Color =  [1 1 1];
% h.Name = 'Value function';
% plot(x, g, 'Color', colo(end, :),'LineWidth',1.5); hold on;
% plot(x, y, 'Color', 'k','LineWidth',1.5); hold off;
% xlabel('x','Fontsize',Fontsize)
% ylabel('f(x)','Fontsize',Fontsize)
% box off

    
N=5;
idxtrain = randsample(n,N);
idxtrain= sort(idxtrain);
% 
N=3;
idxtrain = floor((n-N)/2):floor((n+N)/2);
N = numel(idxtrain);

xtrain = x(idxtrain);
ytrain = y(idxtrain)';

hyp = theta.cov;

m=1000;
fx_0 = NaN(m, n);
fx_1 = NaN(m, n);

meanfun = 0;
type = 'regression';
hyps.ncov_hyp =2; % number of hyperparameters for the covariance function
hyps.nmean_hyp =0; % number of hyperparameters for the mean function
hyps.hyp_lb = -10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
hyps.hyp_ub = 10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
D = 1;

model = gp_regression_model(D, meanfun, kernelfun, regularization, hyps, lb, ub, kernelname);


[mu_y, sigma2_y, ~,~, Sigma2_y]= model.prediction(theta, xtrain, ytrain', xtrain, []);

% D= 1;
% [phi, dphi_dx] = sample_features_GP(theta.cov, D, kernelname,'RRGP');
% phix = phi(xtrain);
% nfeatures = size(phix,2);
% for i =1  
%     w =randn(nfeatures,1);
%     sample_prior = @(x) (phi(x)*w)';
%     noise =  sigma*randn(N,1);
%     K = kernelfun(theta.cov,xtrain,xtrain, 1);
%     update =  @(x) (K\(ytrain - sample_prior(xtrain)'+noise))'*kernelfun(theta.cov,xtrain,x, 0);
%     posterior = @(x) sample_prior(x) + update(x);
%     fx(i, :)=posterior(x);
% end
nfeatures = 3*256;
D = 1;
 
approximation1.decoupled_bases = 0;
approximation1.nfeatures = nfeatures;
approximation1.method = 'RRGP';

approximation2.decoupled_bases = 1;
approximation2.nfeatures = nfeatures;
approximation2.method = 'RRGP';

for i =1:m  
    gs = sample_GP(theta.cov, xtrain, ytrain, model, approximation1);
    fx_0(i, :)=gs(x);
    gs = sample_GP(theta.cov, xtrain, ytrain, model, approximation2);
    fx_1(i, :)=gs(x);
end

% figure();
% plot(fx_rrgp')
% 
% figure();
% plot(fx_ssgp')
% 
[posterior_mean, posterior_variance, ~, ~, Posterior_cov]= model.prediction(theta, xtrain, ytrain', x, model, []);

xticks = [0,0.5,1];
xticks = [x(1), x(end)];
xlim = [x(1), x(end)];

mr = 1;
mc = 3;
legend_pos = [0,1.05];
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
x0 = 0.5;

fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  [1 1 1];
layout = tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');
i = 0;

nexttile();
i=i+1;
%errorshaded(x,posterior_mean, sqrt(posterior_variance), 'Color',  C(1,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold on
plot_gp(x,posterior_mean, posterior_variance,  C(1,:), linewidth); hold on
plot(x,g, 'Color',  C(2,:),'LineWidth', linewidth/2); hold on;
scatter(xtrain, ytrain, 2*markersize, C(2,:), 'filled'); hold off;
set(gca, 'Xlim', xlim, 'Xtick', xticks);
xlabel('$x$')
box off
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
set(gca, 'Fontsize', Fontsize);
ylim = get(gca, 'Ylim');
title('Exact GP')

nexttile();
i=i+1;
% errorshaded(x,mean(fx_0,1), std(fx_0,1), 'Color',  C(1,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold on
plot_gp(x,mean(fx_0,1)', var(fx_0,1)', C(1,:), linewidth); hold on
plot(x,g, 'Color',  C(2,:),'LineWidth', linewidth/2); hold on;
scatter(xtrain, ytrain, 2*markersize, C(2,:), 'filled'); hold off;
set(gca, 'Xlim', xlim, 'Xtick', xticks);
xlabel('$x$')
box off
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
set(gca, 'Ylim', ylim,'Fontsize', Fontsize);
title('Weight-space')

nexttile();

i=i+1;
% errorshaded(x,mean(fx_1,1), std(fx_1,1), 'Color',  C(1,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold on
plot_gp(x,mean(fx_1,1)', var(fx_1,1)', C(1,:),linewidth); hold on
plot(x,g, 'Color',  C(2,:),'LineWidth', linewidth/2); hold on;
scatter(xtrain, ytrain, 2*markersize, C(2,:), 'filled'); hold off;
set(gca, 'Xlim', xlim, 'Xtick', xticks);
xlabel('$x$')
box off
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
set(gca, 'Ylim', ylim,'Fontsize', Fontsize);
title('Decoupled sampling')

figname  = 'GP_sampling_decoupled_bases_vs_weight_space';
folder = [figure_path,figname];
savefig(fig, [folder,'/', figname, '.fig'])
exportgraphics(fig, [folder,'/' , figname, '.pdf']);
exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);



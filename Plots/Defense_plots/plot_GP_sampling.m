%% Sample GP 

clear all
close all
graphics_style_presentation;
currentFile = mfilename( 'fullpath' );
[pathname,~,~] = fileparts( currentFile );
 figure_path = [pathname, '/Figures/'];
savefigures = 1;

rng(13);%12 
n=100; %resolution
lb = 0; ub = 1;
x = linspace(lb, ub,n);

gen_kernelfun = @Matern52_kernelfun;
kernelfun = @Matern52_kernelfun;
kernelname = 'Matern52';
theta.cov = [log(1/10),0];
theta_gen.cov = theta.cov;
theta.mean= 0;
regularization = 'nugget';
Sigma =gen_kernelfun(theta_gen.cov,x,x, true, regularization);

g =  mvnrnd(constant_mean(x,0), gen_kernelfun(theta_gen.cov, x,x, true, regularization)); %generate a function

sigma = 0;
y = g + sigma*randn(1,n); %add measurement noise

graphics_style_presentation;

% h=figure(1);
% h.Color =  [1 1 1];
% h.Name = 'Value function';
% plot(x, g, 'Color', colo(end, :),'LineWidth',1.5); hold on;
% plot(x, y, 'Color', k,'LineWidth',1.5); hold off;
% xlabel('x','Fontsize',Fontsize)
% ylabel('f(x)','Fontsize',Fontsize)
% box off

    
N=5;
idx_data = randsample(n,N);
idx_data= sort(idx_data);
x_data = x(idx_data);
y_data = y(idx_data)';

 
m=1000;
fx = NaN(m, n);

type = 'regression';
D = 1;
hyps.ncov_hyp =2; % number of hyperparameters for the covariance function
hyps.nmean_hyp =1; % number of hyperparameters for the mean function
hyps.hyp_lb = -10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
hyps.hyp_ub = 10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
meanfun = @constant_mean;
model = gp_regression_model(D, meanfun, kernelfun, regularization, hyps, lb, ub, kernelname);

[posterior_mean, posterior_variance, ~,~, Sigma2_y]= model.prediction(theta, x_data, y_data', x_data, []);

approximation.nfeatures = 256;

approximation.method = 'RRGP';
 
[phi, dphi_dx] = sample_features_GP(theta, model, approximation);
phix = phi(x_data);
nfeatures = size(phix,2);
for i =1:m  
    w =randn(nfeatures,1);
    sample_prior = @(x) (phi(x)*w)';
    noise =  sigma*randn(N,1);
    K = kernelfun(theta.cov,x_data,x_data, true, regularization);
    update =  @(x) (K\(y_data - sample_prior(x_data)'+noise))'*kernelfun(theta.cov,x_data,x, true, regularization);
    posterior = @(x) sample_prior(x) + update(x);
    fx(i, :)=posterior(x);
end

approximation.decoupled_bases = 1;
for i =1:m  
    [gs dgsdx]=  sample_GP(theta, x_data, y_data, model, approximation) ;
    fx(i, :)=gs(x);
end

[posterior_mean, posterior_variance, ~, ~, Posterior_cov]= model.prediction(theta, x_data, y_data', x, []);


mr = 1;
mc = 3;
legend_pos = [0.02,1.0];
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  background_color;
layout = tiledlayout(mr,mc, 'TileSpacing', 'compact', 'padding','compact');
i = 0;
yl = [-3,3]; %[-2.5,2.5]
nexttile();
i=i+1;
h1 = plot_gp(x,posterior_mean, posterior_variance,C(1,:),linewidth, 'background', background);
h2 = plot(x,g, 'Color',  C(2,:),'LineWidth', linewidth); hold on;
%errorshaded(x,posterior_mean, sqrt(posterior_variance), 'Color',  C(1,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold on
h3 = plot(x_data, y_data, 'ro', 'MarkerSize', 10, 'color', C(2,:)); hold on;
scatter(x_data, y_data, markersize, C(2,:), 'filled'); hold off;
set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1]);
xlabel('$x$')
box off
set(gca, 'Ylim', yl,'Fontsize', Fontsize);
legend([h1, h2, h3], 'GP posterior', 'True function', 'Data'); 
legend box off 

nexttile();
i=i+1;
plot(x, sample_prior(x),'LineWidth', linewidth, 'Color', C(1,:)); hold on ;
plot(x, update(x),'LineWidth', linewidth,'Color', C(2,:)); hold on ;
plot(x, posterior(x),'LineWidth', linewidth,'Color', C(3,:)); hold off ;
legend('Sample from the prior', 'Update', 'Posterior')
box off;
legend boxoff
set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1],'Fontsize', Fontsize, 'Ylim', yl);
xlabel('$x$')

nexttile();
i=i+1;
h1 = plot_gp(x, mean(fx,1)', var(fx,1)',C(1,:),linewidth, 'background', background);
h2 = plot(x , fx(1:15,:)','color', k, 'LineWidth', linewidth/4); hold on;
h2 = h2(1);
alpha(0.5)
box off
%title('Samples from the posterior distribution')
set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1],'Fontsize', Fontsize, 'Ylim', yl);
xlabel('$x$')
legend([h1, h2], 'Samples distribution', 'Samples from the GP'); 
legend box off 

 
darkBackground(fig,background,[1 1 1])

figname  = 'GP_sampling';
  export_fig(fig, [figure_path,'/' , figname, '.pdf']);
export_fig(fig, [figure_path,'/' , figname, '.png']);


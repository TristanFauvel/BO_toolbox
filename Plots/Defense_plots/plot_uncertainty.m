%% An example of Gaussian process classification

clear all;
close all;
add_gp_module;

currentFile = mfilename( 'fullpath' );
[pathname,~,~] = fileparts( currentFile );
figure_path = [pathname, '/Figures/'];
savefigures = 1;

n=100;

rng(2)
graphics_style_presentation;

colo= othercolor('GnBu7');
modeltype = 'exp_prop'; % or 'laplace'

link = @normcdf;

kernelfun = @ARD_kernelfun;
theta_true.cov= [3;3];

% Generate a function
lb = 0;
ub = 1;
x = linspace(lb, ub,n);
y = mvnrnd(constant_mean(x,0), kernelfun(theta_true.cov, x,x, 'false', 'false'));
y=y-mean(y);

p= link(y);


ntr =10; % Number of data points

i_tr= randsample(n,ntr,'true');
xtrain = x(:,i_tr);
y_tr = y(:, i_tr);
ctrain = p(i_tr)>rand(1,ntr);

x_test = x;
y_test = y;

% GP classification with the correct hyperparameters
theta =theta_true ; % rand(size(theta_true));

regularization = 'nugget';
post = [];

hyps.ncov_hyp =2; % number of hyperparameters for the covariance function
hyps.nmean_hyp =0; % number of hyperparameters for the mean function
hyps.hyp_lb = -10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
hyps.hyp_ub = 10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
D = 1;
meanfun = 0;

kernelname = 'ARD';
condition = [];

model = gp_classification_model(D, meanfun, kernelfun, regularization, hyps, lb, ub, 'classification', link, modeltype, kernelname, condition);


[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc, dvar_muc_dx]= model.prediction(theta, xtrain, ctrain, x_test, post);


nsamps =10000;
samples= mvnrnd(mu_y, Sigma2_y, nsamps);
mu_c_samples = normcdf(samples);



Cst = sqrt(pi*log(2)/2);
h = @(p) -p.*log(p+eps) - (1-p).*log(1-p+eps);

I1 = h(mu_c);
I2 =  log(2)*Cst.*exp(-0.5*mu_y.^2./(sigma2_y+Cst^2))./sqrt(sigma2_y+Cst^2);

%%

N = 100;
sigma2_y_range = linspace(0,10,N);
mu_y_range = linspace(-8,8,N);
h = @(p) -p.*log(p+eps) - (1-p).*log(1-p+eps);

[p,q]= meshgrid(mu_y_range, sigma2_y_range);

inputs  = [p(:),q(:)]';
sigma2_y = inputs(2,:);
mu_y = inputs(1,:);

mu_c = normcdf(mu_y./sqrt(1+sigma2_y));
C = sqrt(pi*log(2)/2);
I1 = h(mu_c);
I2 =  log(2)*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
I = I1 - I2;

h = mu_y./sqrt(1+sigma2_y);
a = 1./sqrt(1+2*sigma2_y);

[tfn_output, dTdh, dTda] = tfn(h, a);
var_muc = (mu_c - 2*tfn_output) - mu_c.^2;
aleatoric_unvar=2*tfn_output;

maxI = max([var_muc, aleatoric_unvar, aleatoric_unvar+var_muc]);
 
%%
 mr = 1;
mc = 3;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth 0.7*fheight(mr)]);
fig.Color =  background_color;
tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');
% nexttile();
% i=i+1;
% imagesc(mu_y_range, sigma2_y_range, reshape(mu_c,N,N)); hold on;
% xlabel('$\mu_f(x)$','Fontsize',Fontsize)
% ylabel('$\sigma^2_f(x)$','Fontsize',Fontsize)
% set(gca,'YDir','normal')
% pbaspect([1 1 1])
% cb = colorbar;
% cb.FontName = 'CMU Serif';
% cb.FontSize = Fontsize;
% colormap(cmap)
% title('$\mu_c(x)$')
% set(gca, 'fontsize', Fontsize)
% text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)

 
nexttile();
i=i+1;
imagesc(mu_y_range, sigma2_y_range, reshape(aleatoric_unvar+var_muc,N,N)); hold on;
xlabel('$\mu_f(x)$','Fontsize',Fontsize)
ylabel('$\sigma^2_f(x)$','Fontsize',Fontsize)

set(gca,'YDir','normal','CLim',[0, maxI])
pbaspect([1 1 1])
% cb = colorbar;
% set(cb, 'Limits', [0, maxI])
% cb.FontName = 'CMU Serif';
% cb.FontSize = Fontsize;
% colormap(cmap)
title('V$(c|x, \mathcal{D})$')
set(gca, 'fontsize', Fontsize)


nexttile();
i=i+1;
imagesc(mu_y_range, sigma2_y_range, reshape(var_muc,N,N)); hold on;
xlabel('$\mu_f(x)$','Fontsize',Fontsize)
% ylabel('$\sigma^2_f(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal','CLim',[0, maxI])
pbaspect([1 1 1])
% cb = colorbar;
title('V$[\Phi(f(x))|\mathcal{D}]$')
set(gca, 'fontsize', Fontsize)


nexttile();
i=i+1;
imagesc(mu_y_range, sigma2_y_range, reshape(aleatoric_unvar,N,N)); hold on;
xlabel('$\mu_f(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal','CLim',[0, maxI])
pbaspect([1 1 1])
 title('E$_f$[V$(c|x, f)|\mathcal{D}]$')
set(gca, 'fontsize', Fontsize)

 cb = colorbar;
set(cb, 'Limits', [0, maxI])
cb.FontName = 'CMU Serif';
cb.FontSize = Fontsize;
colormap(cmap)

maxI = max([I1, I2, I]);
darkBackground(fig,background,[1 1 1])
figname  = 'Uncertainties_var';
export_fig(fig, [figure_path,'/' , figname, '.pdf']);
export_fig(fig, [figure_path,'/' , figname, '.png']);
export_fig(fig, [figure_path,'/' , figname, '.eps']);


 mr = 1;
mc = 3;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth 0.7*fheight(mr)]);
fig.Color =  background_color;
tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');

nexttile();
imagesc(mu_y_range, sigma2_y_range, reshape(I1,N,N)); hold on;
i=i+1;
xlabel('$\mu_f(x)$','Fontsize',Fontsize)
ylabel('$\sigma^2_f(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal','CLim',[0, maxI])
set(gca,'YDir','normal')
pbaspect([1 1 1])
title('H$(c|x)$')
set(gca, 'fontsize', Fontsize)

nexttile();
imagesc(mu_y_range, sigma2_y_range, reshape(I,N,N)); hold on;
i=i+1;
xlabel('$\mu_f(x)$','Fontsize',Fontsize)
% ylabel('$\sigma^2_f(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal','CLim',[0, maxI])
pbaspect([1 1 1])
% cb = colorbar;
title('I$(c,f(x))$')
set(gca, 'fontsize', Fontsize)
% set(get(cb,'Title'),'String','(nats)', 'Interpreter', 'latex')

nexttile();
imagesc(mu_y_range, sigma2_y_range, reshape(I2,N,N)); hold on;
i=i+1;
xlabel('$\mu_f(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal','CLim',[0, maxI])
set(gca,'YDir','normal')
pbaspect([1 1 1])
% cb = colorbar;
title('E$_f[H(c|x, f)]$')
set(gca, 'fontsize', Fontsize)
% set(get(cb,'Title'),'String','(nats)', 'Interpreter', 'latex')

 cb = colorbar;
cb.FontName = 'CMU Serif';
cb.FontSize = Fontsize;
set(cb, 'Limits', [0, maxI])
set(get(cb,'Title'),'String','(nats)', 'Interpreter', 'latex')
 

colormap(cmap)

darkBackground(fig,background,[1 1 1])
figname  = 'Uncertainties_H';
export_fig(fig, [figure_path,'/' , figname, '.pdf']);
export_fig(fig, [figure_path,'/' , figname, '.png']);
export_fig(fig, [figure_path,'/' , figname, '.eps']);
%%


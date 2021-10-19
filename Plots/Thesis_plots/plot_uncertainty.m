%% An example of Gaussian process classification

clear all;
close all;
add_gp_module;

figure_path = '/home/tfauvel/Documents/PhD/Figures/Thesis_figures/Chapter_1/';
n=100;

rng(2)
graphics_style_paper;
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

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
fun = @(x_test) model.prediction(theta, xtrain, ctrain, x_test, post);
dF = test_matrix_deriv(fun, x_test, 1e-8);

%Xlim= [min(x),max(x)];
%Ylim = [-5,5];

legend_pos = [-0.18,1];

mr = 1;
mc = 3;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  [1 1 1];
layout = tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');

i = 0;

nexttile();
i=i+1;
%errorshaded(x,mu_c, sqrt(var_muc), 'Color',  C(1,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold off
plot(x_test, p, 'LineWidth',linewidth,'Color',  C(2,:)) ; hold on;
plot(x_test, mu_c, 'LineWidth',linewidth,'Color', C(1,:)) ; hold on;
scatter(xtrain(ctrain == 1), ctrain(ctrain == 1), markersize, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k') ; hold on;
scatter(xtrain(ctrain == 0), ctrain(ctrain == 0), markersize, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k') ; hold off;
legend('$P(c=1)$', '$\mu_c(x)$','$c=1$', '$c=0$','Fontsize',Fontsize, 'Location', 'northeast')
xlabel('$x$','Fontsize',Fontsize)
legend boxoff
grid off
box off
pbaspect([1 1 1])
set(gca, 'Fontsize', Fontsize);
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)


nexttile();
i=i+1;
p1 = plot_gp(x,mu_y, sigma2_y, C(1,:),linewidth);
p2 = plot(x,y,'LineWidth',linewidth,'Color', C(2,:)); hold off;
% errorshaded(x,mu_y, sqrt(sigma2_y), 'Color',  C(1,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold off
% legend('True function', 'Inferred function','Fontsize',Fontsize)
legend([p1, p2], {'$\mu_f(x)$','$f(x)$'},'Fontsize',Fontsize, 'Location', 'northeast')
xlabel('$x$','Fontsize',Fontsize)
legend boxoff
grid off
box off
pbaspect([1 1 1])
set(gca, 'Fontsize', Fontsize);
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)

nexttile();
i=i+1;
ax1 = imagesc(x, x, Sigma2_y); hold on;
xlabel('$x$','Fontsize',Fontsize)
ylabel('$x''$','Fontsize',Fontsize)
%title('Posterior covariance','Fontsize',Fontsize, 'interpreter', 'latex')
set(gca,'YDir','normal')
pbaspect([1 1 1])
cb = colorbar;
cb.FontName = 'CMU Serif';
cb.FontSize = Fontsize;

cb_lim = get(cb, 'Ylim');
cb_tick = get(cb, 'Ytick');
set(cb,'Ylim', cb_lim, 'Ytick', [cb_tick(1),cb_tick(end)]);
colormap(cmap)
set(gca, 'Fontsize', Fontsize);
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
title('Cov$(f(x),f(x'')|\mathcal{D})$')

figname  = 'GP_classification';
folder = [figure_path,figname];
savefig(fig, [folder,'/', figname, '.fig'])
exportgraphics(fig, [folder,'/' , figname, '.pdf']);
exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);


nsamps =10000;
samples= mvnrnd(mu_y, Sigma2_y, nsamps);
mu_c_samples = normcdf(samples);



 Cst = sqrt(pi*log(2)/2);
    h = @(p) -p.*log(p+eps) - (1-p).*log(1-p+eps);

    I1 = h(mu_c);
    I2 =  log(2)*Cst.*exp(-0.5*mu_y.^2./(sigma2_y+Cst^2))./sqrt(sigma2_y+Cst^2);

    

mr = 3;
mc = 3;
i = 0;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  [1 1 1];
tiledlayout(mr,mc, 'TileSpacing' , 'tight', 'Padding', 'tight')
nexttile()
i=i+1;
plot_gp(x,mu_y, sigma2_y, C(1,:), linewidth);
xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$f(x)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
title('Posterior GP on the latent function','Fontsize', Fontsize)
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
box off
% pbaspect([1 1 1])

nexttile()
i=i+1;
plot(x,mu_c, 'color', C(1,:));
xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$\mu_c(x)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
title('$\mu_c(x)$','Fontsize', Fontsize)
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
box off
% pbaspect([1 1 1])

nexttile()

nexttile()
i=i+1;
 plot(x, I1, 'color', C(1,:))
xlabel('$x$', 'Fontsize', Fontsize)
ylabel('I$(c,f(x)|\mathcal{D})$')
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
box off
title('Total uncertainty ')
set(get(cb,'Title'),'String','(nats)', 'Interpreter', 'latex')


nexttile()

i=i+1;
I= BALD(theta, xtrain, ctrain, x, model,post);
plot(x, I, 'color', C(1,:))
xlabel('$x$', 'Fontsize', Fontsize)
ylabel('I$(c,f(x)|\mathcal{D})$')
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
box off
title('Epistemic uncertainty I$(c,f(x)|\mathcal{D})$')
set(get(cb,'Title'),'String','(nats)', 'Interpreter', 'latex')

nexttile()
i=i+1;
plot(x, I2, 'color', C(1,:))
xlabel('$x$', 'Fontsize', Fontsize)
ylabel('H$(c|\mu_c)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
box off
title('Aleatoric uncertainty $h(\mu_c(c))$')
set(get(cb,'Title'),'String','(nats)', 'Interpreter', 'latex')


nexttile()
i=i+1;
plot(x, var_muc, 'color', C(1,:))
xlabel('$x$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
box off
title('Epistemic uncertainty $V(\Phi(f))$')
 


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
%%

 
legend_pos = [-0.5,1.18];
i=0;
mr = 2;
mc = 3;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth 0.7*fheight(mr)]);
fig.Color =  [1 1 1];
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

maxI = max([var_muc, aleatoric_unvar, aleatoric_unvar+var_muc]);

nexttile();
i=i+1;
imagesc(mu_y_range, sigma2_y_range, reshape(aleatoric_unvar+var_muc,N,N)); hold on;
% xlabel('$\mu_f(x)$','Fontsize',Fontsize)
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
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)


nexttile();
i=i+1;
imagesc(mu_y_range, sigma2_y_range, reshape(var_muc,N,N)); hold on;
% xlabel('$\mu_f(x)$','Fontsize',Fontsize)
% ylabel('$\sigma^2_f(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal','CLim',[0, maxI])
pbaspect([1 1 1])
% cb = colorbar;
title('V$[\Phi(f(x))|\mathcal{D}]$')
set(gca, 'fontsize', Fontsize)
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)


nexttile();
i=i+1;
imagesc(mu_y_range, sigma2_y_range, reshape(aleatoric_unvar,N,N)); hold on;
% xlabel('$\mu_f(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal','CLim',[0, maxI])
pbaspect([1 1 1])
 title('E$_f$[V$(c|x, f)|\mathcal{D}]$')
set(gca, 'fontsize', Fontsize)
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)

 cb = colorbar;
set(cb, 'Limits', [0, maxI])
cb.FontName = 'CMU Serif';
cb.FontSize = Fontsize;
colormap(cmap)

maxI = max([I1, I2, I]);

nexttile();
imagesc(mu_y_range, sigma2_y_range, reshape(I1,N,N)); hold on;
i=i+1;
xlabel('$\mu_f(x)$','Fontsize',Fontsize)
ylabel('$\sigma^2_f(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal','CLim',[0, maxI])
set(gca,'YDir','normal')
pbaspect([1 1 1])
title('H$(c|x, \mathcal{D})$')
set(gca, 'fontsize', Fontsize)
 text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)

nexttile();
imagesc(mu_y_range, sigma2_y_range, reshape(I,N,N)); hold on;
i=i+1;
xlabel('$\mu_f(x)$','Fontsize',Fontsize)
% ylabel('$\sigma^2_f(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal','CLim',[0, maxI])
pbaspect([1 1 1])
% cb = colorbar;
title('I$(c,f(x)|\mathcal{D})$')
set(gca, 'fontsize', Fontsize)
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
% set(get(cb,'Title'),'String','(nats)', 'Interpreter', 'latex')

nexttile();
imagesc(mu_y_range, sigma2_y_range, reshape(I2,N,N)); hold on;
i=i+1;
xlabel('$\mu_f(x)$','Fontsize',Fontsize)
ylabel('$\sigma^2_f(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal','CLim',[0, maxI])
set(gca,'YDir','normal')
pbaspect([1 1 1])
% cb = colorbar;
title('E$_f[H(c|x, f)|\mathcal{D}]$')
set(gca, 'fontsize', Fontsize)
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
% set(get(cb,'Title'),'String','(nats)', 'Interpreter', 'latex')

 cb = colorbar;
cb.FontName = 'CMU Serif';
cb.FontSize = Fontsize;
set(cb, 'Limits', [0, maxI])
set(get(cb,'Title'),'String','(nats)', 'Interpreter', 'latex')
 

figure_path ='/home/tfauvel/Documents/PhD/Figures/Thesis_figures';
figname  = 'Uncertainties';
folder = figure_path;
savefig(fig, [folder,'/', figname, '.fig'])
exportgraphics(fig, [folder,'/' , figname, '.pdf']);
exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);


%%
mr = 1;
mc = 3;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  [1 1 1];
tiledlayout(1,mc, 'TileSpacing', 'tight', 'padding','compact');
nexttile();
imagesc(mu_y_range, sigma2_y_range, reshape(mu_c,N,N)); hold on;
xlabel('$\mu_f(x)$','Fontsize',Fontsize)
ylabel('$\sigma^2_f(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal')
pbaspect([1 1 1])
cb = colorbar;
cb.FontName = 'CMU Serif';
cb.FontSize = Fontsize;
colormap(cmap)
title('$\mu_c(x)$')
set(gca, 'fontsize', Fontsize)
set(get(cb,'Title'),'String','(nats)', 'Interpreter', 'latex')

nexttile();
imagesc(mu_y_range, sigma2_y_range, reshape(I2,N,N)); hold on;
xlabel('$\mu_f(x)$','Fontsize',Fontsize)
ylabel('$\sigma^2_f(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal')
pbaspect([1 1 1])
cb = colorbar;
cb.FontName = 'CMU Serif';
cb.FontSize = Fontsize;
colormap(cmap)
title('E$[h(c|f, \mathcal{D})]$')
set(gca, 'fontsize', Fontsize)
set(get(cb,'Title'),'String','(nats)', 'Interpreter', 'latex')

nexttile();
imagesc(mu_y_range, sigma2_y_range, reshape(aleatoric_unvar,N,N)); hold on;
xlabel('$\mu_f(x)$','Fontsize',Fontsize)
ylabel('$\sigma^2_f(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal')
pbaspect([1 1 1])
cb = colorbar;
cb.FontName = 'CMU Serif';
cb.FontSize = Fontsize;
colormap(cmap)
title('E$[\Phi(f(x))(1-\Phi(f(x)))]$')
set(gca, 'fontsize', Fontsize)
set(get(cb,'Title'),'String','(nats)', 'Interpreter', 'latex')

%%
%%
mr = 1;
mc = 3;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  [1 1 1];
tiledlayout(1,mc, 'TileSpacing', 'tight', 'padding','compact');
nexttile();
imagesc(mu_y_range, sigma2_y_range, reshape(mu_c,N,N)); hold on;
xlabel('$\mu(x)$','Fontsize',Fontsize)
ylabel('$\sigma^2(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal')
pbaspect([1 1 1])
cb = colorbar;
cb.FontName = 'CMU Serif';
cb.FontSize = Fontsize;
colormap(cmap)
title('$\mu_c(x)$')
set(get(cb,'Title'),'String','(nats)', 'Interpreter', 'latex')


nexttile();
imagesc(mu_y_range, sigma2_y_range, reshape(I,N,N)); hold on;
xlabel('$\mu(x)$','Fontsize',Fontsize)
ylabel('$\sigma^2(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal')
pbaspect([1 1 1])
cb = colorbar;
cb.FontName = 'CMU Serif';
cb.FontSize = Fontsize;
colormap(cmap)
title('I$(c,f)$ (nats)')
set(gca, 'fontsize', Fontsize)
set(get(cb,'Title'),'String','(nats)', 'Interpreter', 'latex')

nexttile();
imagesc(mu_y_range, sigma2_y_range, reshape(I1,N,N)); hold on;
xlabel('$\mu(x)$','Fontsize',Fontsize)
ylabel('$\sigma^2(x)$','Fontsize',Fontsize)
set(gca,'YDir','normal')
pbaspect([1 1 1])
cb = colorbar;
colormap(cmap)
cb.FontName = 'CMU Serif';
cb.FontSize = Fontsize;
title('H$(c)$ (nats)')
set(gca, 'fontsize', Fontsize)
set(get(cb,'Title'),'String','(nats)', 'Interpreter', 'latex')

figname  = 'Information';
folder = figure_path;
savefig(fig, [folder,'/', figname, '.fig'])
exportgraphics(fig, [folder,'/' , figname, '.pdf']);
exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);



%%
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc, dvar_muc_dx]= model.prediction(theta, xtrain, ctrain, x, post);

I= BALD(theta, xtrain, ctrain, x, model,post);
% I1 = zeros(N^2,N^2);
% I0 = zeros(N^2,N^2);
dI = zeros(N,N);
for i = 1:N
I1= BALD(theta, [xtrain, x(i)], [ctrain,1], x, model,[]);
I0= BALD(theta, [xtrain, x(i)], [ctrain,0], x, model,[]);
dI(i,:) = I - (mu_c(i)*I1 + (1-mu_c(i))*I0);
end
figure()
imagesc(dI)

figure();
plot(dI); hold on;
plot(I, 'linewidth', linewidth)

i = 50;
I1= BALD(theta, [xtrain, x(i)], [ctrain,1], x, model,[]);
I0= BALD(theta, [xtrain, x(i)], [ctrain,0], x, model,[]);
dI(i,:) = I - (mu_c(i)*I1 + (1-mu_c(i))*I0);

figure()
subplot(2,1,1)
plot(I1); hold on;
plot(I0); hold on;
plot(I)
legend('I1', 'I0', 'I')
subplot(2,1,2)
plot(dI(i,:))

figure()
plot(diag(dI)); hold on;
plot(I);

%%
dIm = zeros(1,N);
for i = 1:N
I1= BALD(theta, [xtrain, x(i)], [ctrain,1], x, model,[]);
I0= BALD(theta, [xtrain, x(i)], [ctrain,0], x, model,[]);
dIm(i) = max(I) - (mu_c(i)*max(I1) + (1-mu_c(i))*max(I0));
end
figure()
plot(dIm); hold on;
plot(diag(dI)); 
plot(I); hold off;

 
clear all
add_gp_module;
figure_path = '/home/tfauvel/Documents/PhD/Figures/Thesis_figures/Chapter_1/';

graphics_style_paper
close all
rng(1)

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

link = @normcdf; %inverse link function

%% Define the range of parameters
n = 100; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%100
lb = 0;
ub = 1;
x = linspace(lb, ub, n);
d =1;
ntr = 5;

[p,q]= meshgrid(x);
x2d = [p(:), q(:)]';

approximation.methodn = 'RRGP';

modeltype = 'exp_prop'; % Approximation method
base_kernelfun =  @Matern52_kernelfun;%kernel used within the preference learning kernel, for subject = computer
base_kernelname = 'Matern52';
theta.cov = [log(1/10),0];
theta.mean= 0;
% 
% base_kernelfun =  @ARD_kernelfun;%kernel used within the preference learning kernel, for subject = computer
% base_kernelname = 'ARD';
% theta = [4,0];
[a,b] = min((x-0.3).^2);
condition.x0 = x(b);
condition.y0 = 0;
x0 = condition.x0;

kernelfun = @(theta, xi, xj, training, regularization) conditional_preference_kernelfun(theta, base_kernelfun, xi, xj, training, 'no', condition.x0);
% kernelfun = @(theta, xi, xj, training) preference_kernelfun(theta, base_kernelfun, xi, xj, training); %%%%%%%%%%%%%%%%%%%%%%%%%%
link = @normcdf; %inverse link function for the classification model

% gfunc = @(x) forretal08(x)/10;
% gfunc = @(x) normpdf(x, 0.5, 0.2);
% g = gfunc(x)-gfunc(x0);
regularization = 'nugget';
g = mvnrnd(zeros(1,n),conditioned_kernelfun(theta.cov, base_kernelfun, x, x, 'false', condition.x0, regularization));

% figure(); plot(g);

f = g'-g;
f= f(:);

nsamp= 5;
rd_idx = randsample(size(x2d,2), nsamp, 'true');
xtrain= x2d(:,rd_idx);
ytrain= f(rd_idx);
ctrain = link(ytrain)>rand(nsamp,1);

regularization = 'nugget';
model.condition = condition;
model.base_kernelfun = base_kernelfun;
D  = 1;
kernelname = base_kernelname;
meanfun=0;
hyps.ncov_hyp =2; % number of hyperparameters for the covariance function
hyps.nmean_hyp =0; % number of hyperparameters for the mean function
hyps.hyp_lb = -10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
hyps.hyp_ub = 10*ones(hyps.ncov_hyp  + hyps.nmean_hyp,1);
type = 'preference';
model = gp_classification_model(D, meanfun, kernelfun, regularization, hyps, ...
    lb, ub, type, link, modeltype, kernelname, condition);

 post = [];

[mu_c,  mu_f, sigma2_f, Sigma2_f] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), x2d, post);

[~,  mu_g, sigma2_g, Sigma2_g ,~, ~, ~, ~, ~, ~, post] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), [x; x0*ones(1,n^d)], post);

mu_g = -mu_g; %(because prediction_bin considers P(x1 > x2);



%% Find the true global optimum of g
[gmax, id_xmax] = max(g);
xmax = x(id_xmax);
 
nsamps = 1000;
samples_g = NaN(nsamps, n);
samples_prior = NaN(nsamps, n);
samples_f =  NaN(nsamps, n*n);
updates = NaN(nsamps, n);
D = 1;
nfeatures = 128;
 
approximation.method= 'RRGP';
approximation.nfeatures = nfeatures;
approximation.decoupled_bases = 1;

if  strcmp(type, 'preference')
    [approximation.phi_pref, approximation.dphi_pref_dx, approximation.phi, approximation.dphi_dx]= sample_features_preference_GP(theta, d, model, approximation);
else
    [approximation.phi, approximation.dphi_dx]= sample_features_GP(theta, model, approximation);
end


for j = 1:nsamps
    [sample_f, samples_g(j,:), decomposition] = sample_preference_GP(x, theta, xtrain, ctrain, model, approximation, post);
    samples_f(j,:) = sample_f;
end

cl_mean = [min([mu_f;mean(samples_f)']), max([mu_f;mean(samples_f)'])];
cl_sigma  = [min([sigma2_f;var(samples_f)']), max([sigma2_f;var(samples_f)'])];


mr =2;
mc = 3;

fig=figure('units','centimeters','outerposition',1+[0 0 16 fheight(mr)]);
fig.Color =  background_color;
legend_pos = [-0.2,1.15];

i = 0;
% tiledlayout(mr,mc, 'TileSpacing' , 'tight', 'Padding', 'tight')
tiledlayout(mr,mc, 'TileSpacing' , 'tight', 'Padding', 'compact')

nexttile(1);
i=i+1;
plot(x, decomposition.sample_prior(x),'LineWidth', linewidth); hold on ;
plot(x, decomposition.update_1(x),'LineWidth', linewidth); hold on ;
plot(x, decomposition.cond_sample_prior(x),'LineWidth', linewidth); hold on ;
vline(condition.x0); hold on;
plot([0,1], [condition.y0,condition.y0], '--k'); hold off;
legend('Prior', 'Update', 'Posterior')
box off;
legend boxoff
xlabel('$x$')
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
set(gca, 'Xlim', [0,1], 'Xtick', [0,0.5,1],'Fontsize', Fontsize);
pbaspect([1 1 1])


nexttile(4);
i=4;
imagesc(x, x, reshape(sample_f, n,n), cl_mean); hold on;
scatter(xtrain(1, ctrain(1:ntr)==1),xtrain(2, ctrain(1:ntr)==1), markersize, 'o', 'k','filled'); hold on;
scatter(xtrain(1, ctrain(1:ntr)==0),xtrain(2, ctrain(1:ntr)==0), markersize, 'o','k'); hold off;
title('$\mu_f(x,x'')$')
set(gca,'YDir','normal', 'Fontsize', Fontsize)
ylabel('$x''$')
cb = colorbar;
cb.FontName = 'CMU Serif';
cb.FontSize = Fontsize;
set(gca,'XTick',[0 0.5 1],'YTick',[0 0.5 1])
pbaspect([1 1 1])
colormap(cmap)
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)


nexttile(2)
i=2;
imagesc(x, x, reshape(mu_f, n,n), cl_mean); hold on;
scatter(xtrain(1, ctrain(1:ntr)==1),xtrain(2, ctrain(1:ntr)==1), markersize, 'o', 'k','filled'); hold on;
scatter(xtrain(1, ctrain(1:ntr)==0),xtrain(2, ctrain(1:ntr)==0), markersize, 'o','k'); hold off;
title('$\mu_f(x,x'')$')
set(gca,'YDir','normal', 'Fontsize', Fontsize)
ylabel('$x''$')
cb = colorbar;
cb.FontName = 'CMU Serif';
cb.FontSize = Fontsize;
set(gca,'XTick',[0 0.5 1],'YTick',[0 0.5 1])
pbaspect([1 1 1])
colormap(cmap)
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)

nexttile(3)
i=3;
imagesc(x, x, reshape(sigma2_f, n,n), cl_sigma); hold on;
scatter(xtrain(1, ctrain(1:ntr)==1),xtrain(2, ctrain(1:ntr)==1), markersize, 'o', 'k','filled'); hold on;
scatter(xtrain(1, ctrain(1:ntr)==0),xtrain(2, ctrain(1:ntr)==0), markersize, 'o','k'); hold off;
title('$\sigma^2_f(x,x'')$')
set(gca,'YDir','normal')
set(gca,'XTick',[0 0.5 1],'YTick',[0 0.5 1], 'Fontsize', Fontsize)
pbaspect([1 1 1])
colormap(cmap)
cb = colorbar;
cb.FontName = 'CMU Serif';
cb.FontSize = Fontsize;

text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)

% 
% nexttile
% i=i+1;
% plot(x, decomposition.cond_sample_prior(x),'LineWidth', linewidth); hold on ;
% plot(x, decomposition.update_2([x;x0.*ones(D,size(x,2))]),'LineWidth', linewidth); hold on ;
% plot(x, samples_g(j,:),'LineWidth', linewidth); hold on ;
% vline(condition.x0); hold on;
% plot([0,1], [condition.y0,condition.y0], '--k'); hold off;
% legend('Prior', 'Update', 'Posterior')
% box off;
% legend boxoff
% xlabel('$x$')
% set(gca, 'Fontsize', Fontsize)
% text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
% pbaspect([1 1 1])
% imagesc(x, x, reshape(sample_f, n,n)); hold on;
% scatter(xtrain(1, ctrain(1:ntr)==1),xtrain(2, ctrain(1:ntr)==1), markersize, 'o', 'k','filled'); hold on;
% scatter(xtrain(1, ctrain(1:ntr)==0),xtrain(2, ctrain(1:ntr)==0), markersize, 'o','k'); hold off;
% % xlabel('$x$')
% % ylabel('$x''$')
% ylabel('$x''$')
% title('$\tilde{f}(x,x'')$')
% set(gca,'YDir','normal')
% set(gca,'XTick',[0 0.5 1],'YTick',[0 0.5 1])
% pbaspect([1 1 1])
% colormap(cmap)

nexttile(5)
i=5;
imagesc(x, x, reshape(mean(samples_f,1), n,n),cl_mean); hold on;
scatter(xtrain(1, ctrain(1:ntr)==1),xtrain(2, ctrain(1:ntr)==1), markersize, 'o', 'k','filled'); hold on;
scatter(xtrain(1, ctrain(1:ntr)==0),xtrain(2, ctrain(1:ntr)==0), markersize, 'o','k'); hold off;
%xlabel('$x$')
title('$\tilde{\mu}_f(x,x'')$')
set(gca,'YDir','normal')
set(gca,'XTick',[0 0.5 1],'YTick',[0 0.5 1], 'Fontsize', Fontsize)
pbaspect([1 1 1])
colormap(cmap)
cb = colorbar;
cb.FontName = 'CMU Serif';
cb.FontSize = Fontsize;
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
xlabel('$x$')

nexttile(6)
i=6;
imagesc(x, x, reshape(var(samples_f,1), n,n),cl_sigma); hold on;
scatter(xtrain(1, ctrain(1:ntr)==1),xtrain(2, ctrain(1:ntr)==1), markersize, 'o', 'k','filled'); hold on;
scatter(xtrain(1, ctrain(1:ntr)==0),xtrain(2, ctrain(1:ntr)==0), markersize, 'o','k'); hold off;
% xlabel('$x$')
% ylabel('$x''$')
title('$\tilde{\sigma}^2_f(x,x'')$')
set(gca,'YDir','normal')
set(gca,'XTick',[0 0.5 1],'YTick',[0 0.5 1], 'Fontsize', Fontsize)
pbaspect([1 1 1])
colormap(cmap)
cb = colorbar;
cb.FontName = 'CMU Serif';
cb.FontSize = Fontsize;
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
xlabel('$x$')
colororder(C)



figname  = 'preference_sampling_GP';
folder = [figure_path,figname];
savefig(fig, [folder,'/', figname, '.fig'])
exportgraphics(fig, [folder,'/' , figname, '.pdf']);
exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);

%%
fig =figure()
options.handle = fig;
options.alpha = 0.3;
options.line_width = linewidth;
options.color_area = cmap(1,:);%[23, 20, 196]./255;    % Blue theme
options.color_line = cmap(1,:);%[23, 20, 196]./255;
options.x_axis = x;
options.line_width = linewidth;
h1 = plot_area(mean(-samples_g), sqrt(var(-samples_g)), options); hold on;
options.color_area = cmap(end,:);%[23, 20, 196]./255;    % Blue theme
options.color_line = cmap(end,:);%[23, 20, 196]./255;
h2 = plot_area(mu_g', sqrt(sigma2_g'), options); hold off;
xlabel('$x$', 'Fontsize', Fontsize)
% ylabel('$g(x)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1])
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3))

text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
box off
xlabel('$x$')
legend([h1,h2], 'Samples distribution', '$P(g(x) | \mathcal{D})$')
legend boxoff


crosscov= cov(samples_f);
h= figure();
h.Color =  [1 1 1];
subplot(1,2,1)
imagesc(crosscov)
title('Sample covariance')
pbaspect([1,1,1])
cb = colorbar;
cb.FontName = 'CMU Serif';
cb.FontSize = Fontsize;
subplot(1,2,2)
imagesc(Sigma2_f)
pbaspect([1,1,1])
title('True covariance')
colorbar


h= figure();
h.Color =  [1 1 1];
imagesc(crosscov-Sigma2_f)
pbaspect([1,1,1])
colorbar


[sample_g, dsample_g_dx, decomposition] = sample_value_GP(theta, xtrain, ctrain, model, approximation, post);


dF = NaN(1,n);
for i= 1:n
    dF(i) = test_matrix_deriv(sample_g, x(i), 1e-12);
    test(i) = dsample_g_dx(x(i));
end
figure();
plot(dF); hold on;
plot(test); hold off;


clear all
add_bo_module;
graphics_style_paper;
figure_path = '/home/tfauvel/Documents/PhD/Figures/Thesis_figures/Chapter_1/';

close all
rng(4)

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

link = @normcdf; %inverse link function

n = 50;
x = linspace(0,1, n);
[p,q]= meshgrid(x);
x2d = [p(:), q(:)]';
n = 3000;
x = linspace(0,1, n);

d =1;
ntr = 80; %100

x0 =0;

modeltype = 'exp_prop'; % Approximation method
base_kernelfun =  @Matern52_kernelfun;%kernel used within the preference learning kernel, for subject = computer
kernelname = 'Matern52';

link = @normcdf; %inverse link function for the classification model
model.regularization = 'nugget';
model.base_kernelfun = base_kernelfun;
model.kernelfun = @(theta, xi, xj, training, reg) conditional_preference_kernelfun(theta, base_kernelfun, xi, xj, training, reg,x0);
cond_base_kernelfun = @(theta, xi, xj, training, reg) conditioned_kernelfun(theta, base_kernelfun, xi, xj, training, x0, reg);

model.link = link;
model.modeltype = modeltype;
model.kernelname = kernelname;
model.lb_norm  = 0;
model.ub_norm  = 1;
model.max_x = [1;1];
model.min_x = [0;0];
model.lb = 0;
model.ub = 1;
model.ns = 0;
model.task = 'max';
model.type = 'preference';
model.condition.x0 = 0;
model.condition.y0 = 0;

post = [];
model.D = 1;
regularization = 'none';
theta= [-1;1];
g = -0.5*mvnrnd(zeros(1,n),cond_base_kernelfun(theta, x, x, 'false', regularization));

f = g-g';
f= f(:);

D=1;
if strcmp(model.kernelname, 'Matern52') || strcmp(model.kernelname, 'Matern32') %|| strcmp(model.kernelname, 'ARD')
    approximation.method = 'RRGP';
else
    approximation.method = 'SSGP';
end
approximation.nfeatures = 4096;
approximation.decoupled_bases= 1;
[approximation.phi_pref, approximation.dphi_pref_dx, approximation.phi, approximation.dphi_dx]= sample_features_preference_GP(theta, D, model, approximation);

figure()
subplot(1,2,1)
plot(g);
subplot(1,2,2)
plot(normcdf(g))

rng(2) %2
nsamp= 1000;
rd_idx = randsample(size(x2d,2), nsamp, 'true');
xtrain= x2d(:,rd_idx);
ytrain= f(rd_idx)';
ctrain = link(ytrain)>rand(nsamp,1);

xtrain = xtrain(:,1:ntr);
ctrain = ctrain(1:ntr);
ytrain = ytrain(1:ntr);

post = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), [], model, post);

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), [x;x0*ones(1,n)], model, post);

[max_muy, idx1] = max(mu_y);
x1 = x(idx1);
[~, xMUC] = maxvar_challenge(theta, xtrain(:,1:ntr), ctrain(1:ntr), model, post, approximation);
rng(1)
[~, xTS] = Thompson_challenge(theta, xtrain(:,1:ntr), ctrain(1:ntr), model, post, approximation);
[~, xBrochu] = Brochu_EI(theta, xtrain(:,1:ntr), ctrain(1:ntr), model, post, approximation);
[~, xBEI] = bivariate_EI(theta, xtrain(:,1:ntr), ctrain(1:ntr), model, post, approximation);
[~, xUCB] = Dueling_UCB(theta, xtrain(:,1:ntr), ctrain(1:ntr), model, post, approximation);


model2 = model;
model2.type = 'classification';
model2.kernelfun = base_kernelfun;
sample_g = sample_GP(theta,  x, g, model2, approximation);


[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), [x;x1*ones(1,n)], model, post);

xx = mu_c;
yy = sqrt(var_muc);
 
 GP = [xx, yy]';
not_pareto = [];
idx_pareto = [];
pareto_front = [];
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


legend_pos = [-0.15,1];
Y = normcdf(mvnrnd(mu_y,Sigma2_y,5000));


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

xp = NaN(1,n); 
yp = NaN(1,n);
xp(idx_pareto) = x(idx_pareto);
yp(idx_pareto) = normcdf(g(idx_pareto)-sample_g(x1));
p3 = plot(xp, yp, 'color', C(2,:), 'linewidth', linewidth);  hold on;
xp = NaN(1,n); 
yp = NaN(1,n);
xp(setdiff(1:n,idx_pareto)) = x(setdiff(1:n,idx_pareto));
yp(setdiff(1:n,idx_pareto)) = normcdf(g(setdiff(1:n,idx_pareto))- sample_g(x1));
p4 = plot(xp, yp, 'color',  0.5*[1 1 1], 'linewidth', linewidth);  hold on;

h2 = scatter(xMUC, normcdf(sample_g(xMUC)-sample_g(x1)), 10*markersize, 'b', '+','LineWidth',1.5); hold on;
h3 = scatter(xTS, normcdf(sample_g(xTS)-sample_g(x1)), 10*markersize, 'm','x','LineWidth',1.5); hold on;
h4 = scatter(xBrochu, normcdf(sample_g(xBrochu)-sample_g(x1)), 10*markersize,'g', 'x','LineWidth',1.5); hold on;
h5 = scatter(xBEI, normcdf(sample_g(xBEI)-sample_g(x1)), 10*markersize, 'x','LineWidth',1.5); hold on;
h6 = scatter(xUCB, normcdf(sample_g(xUCB)-sample_g(x1)), 10*markersize, 'c', 'o','LineWidth',1.5); hold on;

[xt,b] = sort([0,0.5, x1, 1]);
xticks(xt)
lgs = {'0', '0.5', '$x_1$','1'};
xticklabels(lgs(b))

xlabel('$x$', 'Fontsize', Fontsize)
ylabel('$P(x>x_1)$', 'Fontsize', Fontsize)
set(gca,'XTick',[0 0.5 1],'Fontsize', Fontsize)
ytick = get(gca,'YTick');
set(gca,'YTick', linspace(min(ytick), max(ytick), 3), 'Fontsize', Fontsize)
%title('Inferred value function $g(x)$','Fontsize', Fontsize)
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
box off
% pbaspect([1 1 1])
legend([p3, p2, p1], '$P(x>x_1)$, Pareto front', '$P(x>x_1|\mathcal{D})$', '$\mu_c(x,x_1)$')
legend box off
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)


nexttile();
i=i+1;
% s = scatter(not_pareto(1,:),not_pareto(2,:), markersize, 'k', 'filled'); hold on;
h1 = plot(pareto_front(1,:),pareto_front(2,:),'color', C(2,:), 'linewidth', linewidth); hold on;
h2 = plot(not_pareto(1,:),not_pareto(2,:),'color', 0.5*[1 1 1], 'linewidth', linewidth); hold on;

box off
xlabel('E$[\Phi(f(x, x_1))|\mathcal{D}]$')
ylabel('V$[\Phi(f(x, x_1))|\mathcal{D}]$')
set(gca, 'Fontsize', Fontsize);

%%

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), [xMUC; x1], model, post);
h2 = scatter(mu_c, sqrt(var_muc), 10*markersize,'b', '+','LineWidth',1.5); hold on;

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), [xTS; x1], model, post);
h3 = scatter(mu_c, sqrt(var_muc), 10*markersize, 'm','x','LineWidth',1.5); hold on;

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), [xBrochu; x1], model, post);
h4 = scatter(mu_c, sqrt(var_muc), 10*markersize,'g', 'x','LineWidth',1.5); hold on;

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), [xBEI; x1], model, post);
h5 = scatter(mu_c, sqrt(var_muc), 10*markersize, 'x','LineWidth',1.5); hold on;

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), [xUCB; x1], model, post);
h6 = scatter(mu_c, sqrt(var_muc), 10*markersize,'c', 'o','LineWidth',1.5); hold on;

text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)

legend([h1 h2 h3 h4 h5 h6], 'Pareto front', 'MUC', 'Dueling Thompson', 'EI', 'Bivariate EI', 'Dueling UCB');
legend box off
s.AlphaData = 0.5;
s.MarkerFaceAlpha = 'flat';
 
figname  = 'Pareto_front_PBO';
folder = [figure_path,figname];
savefig(fig, [folder,'/', figname, '.fig'])
exportgraphics(fig, [folder,'/' , figname, '.pdf']);
exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);


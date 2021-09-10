clear all
add_bo_module;
graphics_style_paper;
figure_path = '/home/tfauvel/Documents/PhD/Figures/Thesis_figures/Chapter_1/';

close all
rng(1)

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

link = @normcdf; %inverse link function

%% Define the range of parameters
n = 50;
x = linspace(0,1, n);
d =1;
ntr = 5;

[p,q]= meshgrid(x);
x2d = [p(:), q(:)]';

x0 = x(:,1);

modeltype = 'exp_prop'; % Approximation method
base_kernelfun =  @Matern52_kernelfun;%kernel used within the preference learning kernel, for subject = computer
base_kernelname = 'Matern52';
approximationimation = 'RRGP';
condition.x0 = x0;
condition.y0 = 0;

kernelfun = @(theta, xi, xj, training, regularization) conditional_preference_kernelfun(theta, base_kernelfun, xi, xj, training,regularization, condition.x0);
link = @normcdf; %inverse link function for the classification model
model.regularization = 'nugget';
model.kernelfun = kernelfun;
model.base_kernelfun = base_kernelfun;

model.link = link;
model.modeltype = modeltype;
model.kernelname = base_kernelname;
model.condition = condition;
post = [];
model.D = 1;
% gfunc = @(x) forretal08(x)/10;
% gfunc = @(x) normpdf(x, 0.5, 0.2);
% g = gfunc(x)-gfunc(x0);

regularization = 'nugget';
theta= [-1;1];
g = mvnrnd(zeros(1,n),base_kernelfun(theta, x, x, 'false', regularization));
g = g-g(1);

f = g'-g;
f= f(:);

nsamp= 500;
rd_idx = randsample(size(x2d,2), nsamp, 'true');
xtrain= x2d(:,rd_idx);
ytrain= f(rd_idx);
ctrain = link(ytrain)>rand(nsamp,1);


[mu_c,  mu_f, sigma2_f] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), x2d, model, post);
[~,  mu_g, sigma2_g, Sigma2_g, ~,~,~,~,~,~,post] = prediction_bin(theta, xtrain(:,1:ntr), ctrain(1:ntr), [x; x0*ones(1,n^d)], model, post);
mu_g = -mu_g; %(because prediction_bin considers P(x1 > x2);



%% Find the true global optimum of g
[gmax, id_xmax] = max(g);
xmax = x(id_xmax);
mr = 2;
mc = 3;


mr = 1;
mc = 1;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(1)]);
fig.Color =  [1 1 1];
layout1 = tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');
i = 0;
Xlim = [0,1];
nexttile();
h1 = plot(pareto_front(1,:),pareto_front(2,:),'color', C(1,:), 'linewidth', linewidth);
box off
xlabel('$\mu(x)$')
ylabel('$\sigma(x)$')
legend([h1], 'Pareto front')
legend box off
set(gca, 'Fontsize', Fontsize);

figname  = 'Pareto_front';
folder = [figure_path,figname];
savefig(fig, [folder,'/', figname, '.fig'])
exportgraphics(fig, [folder,'/' , figname, '.pdf']);
exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);


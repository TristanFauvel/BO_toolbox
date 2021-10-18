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
g = -0.5*mvnrnd(zeros(1,n),base_kernelfun(theta, x, x, 'false', regularization));

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

post = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), [], post);

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx, dSigma2y_dx, var_muc] = model.prediction(theta, xtrain(:,1:ntr), ctrain(1:ntr), [x;x0*ones(1,n)], post);

[max_muy, idx1] = max(mu_y);
x1 = x(idx1);

nsamples = 3;
KSSx = kernelselfsparring_tour(theta, xtrain_norm, ctrain, model, post, approximation, nsamples)
MVTx = kernelselfsparring_tour(theta, xtrain_norm, ctrain, model, post, approximation, nsamples)

  

  
figname  = 'batch_PBO';
folder = [figure_path,figname];
savefig(fig, [folder,'/', figname, '.fig'])
exportgraphics(fig, [folder,'/' , figname, '.pdf']);
exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);


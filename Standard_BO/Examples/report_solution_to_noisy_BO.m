clear all
add_bo_module
graphics_style_paper
figure_path = '/home/tfauvel/Documents/PhD/Figures/Thesis_figures/Chapter_1/';
close all


letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
legend_pos = [-0.05,1];

%% Define the range of parameters
n = 100;
x = linspace(0,1, n);
d =1;


kernel_name = 'ARD_wnoise';


ncov_hyp=2;
kernelfun = @ARD_kernelfun_wnoise;
theta_true = [5,7,5];%[3,2];

theta.cov = theta_true; %%%%%%%%%%%%%%%%%%%% Known hyperparameters of the covariance

meanfun= @constant_mean;
nmean_hyp = 1;
theta.mean = zeros(nmean_hyp,1);
lb = [-10,0, -5] ; % lengthscale, k0, mu
ub = [10,100, 5] ;

c= othercolor('GnBu7');

s = 1;
rng(s)
%generate a function
g = mvnrnd(meanfun(x,0), kernelfun(theta_true, x,x,false));
g = g- mean(g);

nd = 1;
sigma2 = exp(theta_true(nd+2));
y = g + sqrt(sigma2)*randn(1,n);

x_test = x;
y_test = y;


maxiter=30;
s = rng;

idx= randsample(n,maxiter); % for random sampling


if numel(idx)==1
    i_tr = idx;
else
    i_tr= randsample(idx,1);
end
new_x = x(:,i_tr);
new_y = y(:,i_tr);

x_tr = [];
y_tr = [];
regularization = 'nugget';
for i =1:3
    x_tr = [x_tr, new_x];
    y_tr = [y_tr, new_y];
    
   
    [mu_y, sigma2_y, ~, ~, Sigma2_y]= prediction(theta, x_tr, y_tr, x, kernelfun, meanfun, [], regularization);
    i_tr= idx(i); %random sampling
    new_x = x(:,i_tr);
    new_y = y(:,i_tr); % no noise %%%%%%%%%%%%%%%%%%%
    
        
end
sigma2_y(sigma2_y<0)=0;

%% Compute the max among the training set
[bestf, ibestf] = max(y_tr);
xbestf = x(:,ibestf);
%% Compute the max of the posterior mean
[max_mu, imax_mu]= max(mu_y);
xmax_mu= x(:,imax_mu);
%% Compute the probability of being the max
nsamps = 100000;
samps = mvnrnd(mu_y, Sigma2_y, nsamps);
[a,b]= max(samps,[],2);

for i = 1:nsamps
    index=find(samps(i,:)==a(i));
    if numel(index)>1
        b(i) = randsample(index);
    end
end


bx = x(b);

[N,edges] = histcounts(bx, n, 'Normalization','probability');
edges = edges(2:end) - (edges(2)-edges(1))/2;
figure()
plot(edges, N);


mr = 1;
mc = 1;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  [1 1 1];
% layout1 = tiledlayout(mr,mc, 'TileSpacing', 'tight', 'padding','compact');
i = 0;

Xlim = [0,1];
i=i+1;
yyaxis left
plot_gp(x,mu_y, sigma2_y, C(1,:), linewidth); hold on
plot(x, g, '-', 'Color',  C(2,:),'LineWidth', linewidth); hold on;
% errorshaded(x,mu_y, sqrt(sigma2_y), 'Color',  C(1,:),'LineWidth', linewidth, 'Fontsize', Fontsize); hold on
vline(xmax_mu,'Linewidth',linewidth, 'ymax', max_mu, 'Color', C(1,:)); hold on;

plot(x_tr, y_tr, 'ro', 'MarkerSize', 10, 'color', C(2,:)); hold on;
scatter(x_tr, y_tr, markersize, C(2,:), 'filled'); hold on;
set(gca, 'Fontsize', Fontsize, 'Xlim', Xlim); %,  'Ylim',Ylim)
grid off
box off
set(gca, 'Xtick', [0,1])
ylabel('$f(x)$','Fontsize',Fontsize)

[a,b]= max(N);
yyaxis right
plot(edges, N,'Color',  'k','LineWidth', linewidth); hold on;
vline(edges(b),'Linewidth',linewidth, 'ymax', N(b)); hold off;
xlabel('$x$','Fontsize',Fontsize)
ylabel('$P(x|\mathcal{D})$','Fontsize',Fontsize)
set(gca, 'Fontsize', Fontsize, 'Xlim', Xlim)
grid off
box off
xticks([0,1])
% xticklabels({'0', '$x_{t+1}','1'})
xticklabels({'0', '1'})
% text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
ax = gca;
ax.YAxis(1).Color = 'k';
ax.YAxis(2).Color = 'k';


figname  = 'BO_solution_wnoise';
folder = [figure_path,figname];
savefig(fig, [folder,'\', figname, '.fig'])
exportgraphics(fig, [folder,'\' , figname, '.pdf']);
exportgraphics(fig, [folder,'\' , figname, '.png'], 'Resolution', 300);



clear all
close all

add_bo_module
graphics_style_paper

n=1000;
rng(9)

update = 'none';
mean_name = 'constant';
kernel_name = 'Gaussian';

if strcmp(kernel_name, 'ARD_wnoise')
    %ARD kernel with noise
    ncov_hyp=3;
    kernelfun = @ARD_kernelfun_wnoise;
    theta_true = [1;2;0];
    
elseif strcmp(kernel_name, 'ARD')
    %ARD kernel
    ncov_hyp=2;
    kernelfun = @ARD_kernelfun;
    theta_true = [1;2];
elseif strcmp(kernel_name, 'Gaussian')
    ncov_hyp=2;
    kernelfun = @Gaussian_kernelfun;
    theta_true = [1;2];
elseif strcmp(kernel_name, 'Gaussian_wnoise')
    ncov_hyp=3;
    nmean_hyp=1;
    kernelfun= @Gaussian_kernelfun_wnoise;
end
theta.cov = rand(ncov_hyp,1);
%%
theta.cov = theta_true; %%%%%%%%%%%%%%%%%%%% Known hyperparameters of the covariance

if strcmp(mean_name, 'constant')
    meanfun= @constant_mean;
    nmean_hyp = 1;
    theta.mean = zeros(nmean_hyp,1);
    lb = [-10,0, -5] ; % lengthscale, k0, mu
    ub = [10,100, 5] ;
elseif strcmp(mean_name, 'gaussian')
    nmean_hyp = 3;
    mu=1.9;
%     mu=3;
    sigma= log(1); 
    alpha=log(2*sqrt(2*pi)*exp(sigma));
    mean_hyp=[mu;sigma;alpha]; %[mean of the gaussian, std]; initial mean hyperparameters of the GP
    theta.mean = mean_hyp;
%     sample_mean_hyp=[0,0.1,10]; %[mean of the gaussian, std];
    nx = 1;
    ny=1;
    meanfun= @(t, hyperparameters) gaussian_prior_mean_1d(t, hyperparameters);    
    lb = [-10,0, 0, -10, 0] ; % lengthscale, k0, mu, sigma, alpha
    ub = [10,100, 5, 10,10] ;
end

if strcmp(update, 'mean') || strcmp(update, 'none')
    lb(1:ncov_hyp) = theta.cov;
    ub(1:ncov_hyp) = theta.cov;
elseif strcmp(update, 'cov') || strcmp(update, 'none')
    lb(ncov_hyp+1:nmean_hyp) = theta.mean;
    ub(ncov_hyp+1:nmean_hyp) = theta.mean;
    
end

        
c= othercolor('GnBu7');

x = linspace(0,5,n);
y = mvnrnd(constant_mean(x,0), kernelfun(theta_true, x,x)); %generate a function

x_test = x;
y_test = y;


maxiter=30;
nopt =0; %set nopt to maxiter +1 to get random acquisition

F(maxiter) = struct('cdata',[],'colormap',[]);
idx= randsample(n,maxiter); % for random sampling

%% Plot the prior
mu_y= meanfun(x_test, theta.mean);
sigma2_y = diag(kernelfun(theta.cov, x_test,x_test));

Xlim =[0,5];

h=figure(21);
h.Name = 'Prior';
h.Color =  [1 1 1];
errorshaded(x, mu_y, sqrt(sigma2_y), 'Color',  c(22,:),'LineWidth', 1.5, 'DisplayName','Prediction', 'Fontsize', 14, ...
    'Xlim', [0,5], 'Opacity', 1); hold on
plot(x,mu_y,'LineWidth',1.5, 'Color', 'k'); hold on;
plot(x,y,'LineWidth',1.5, 'Color', c(end,:)); hold off;

xlabel('x','Fontsize',Fontsize)
ylabel('f(x)','Fontsize',Fontsize)
xlim([0,5])
grid off
box off

cum_regret_i =0;
cum_regret=NaN(1, maxiter+1);
cum_regret(1)=0;

Ylim =  [-5,5];

% i_tr= randsample(n,1);

idx = find(mu_y == max(mu_y));
if numel(idx)==1
    i_tr = idx;
else
    i_tr= randsample(idx,1);
end
new_x = x(:,i_tr);
new_y = y(:,i_tr);

x_tr = [];
y_tr = [];


for i =1:maxiter
    x_tr = [x_tr, new_x];
    y_tr = [y_tr, new_y];
    
    %cum_regret_i  =cum_regret_i + max(y)-new_y;
    cum_regret_i  =cum_regret_i + max(y)-max(y_tr);
    cum_regret(i+1) = cum_regret_i;
    
    [mu_y, sigma2_y]= prediction(theta, x_tr, y_tr, x_test, kernelfun, meanfun);
    
    
    if i> nopt
        options=[];
        hyp=[theta.cov; theta.mean];

        %% Multistart optimization of theta
        ncandidates = 10;
        if strcmp(update, 'mean') || strcmp(update, 'cov') || strcmp(update, 'all')
            fun = @(hyp)minimize_negloglike(hyp, x_tr, y_tr, kernelfun, meanfun, ncov_hyp, nmean_hyp, update);
            hyp = multistart_minfunc(fun, lb, ub, ncandidates, options);
        end
        %%
        theta.cov = hyp(1:ncov_hyp);
        theta.mean = hyp(ncov_hyp+1:ncov_hyp+nmean_hyp);
         
        sigma2_y(sigma2_y<0)=0;
        bestf = max(y_tr);
        EI  = expected_improvement(mu_y, sigma2_y,[], [], bestf, 'max');
        [max_EI, new_i] = max(EI);
        new_x= x(:, new_i);
        new_y = y(x==new_x);
        
        
        h=figure(3);
        h.Name = 'Bayesian optimisation before hyperparameters optimization';
        h.Color =  [1 1 1];
        plot(x,EI,'LineWidth',1.5,'Color', c(end,:)); hold off;
        xlabel('x','Fontsize',Fontsize)
        ylabel('Expected improvement','Fontsize',Fontsize)
        set(gca, 'Fontsize', Fontsize, 'Xlim', Xlim)
        grid off
        box off
        
        h=figure(4);
        h.Name = 'Bayesian optimisation before hyperparameters optimization';
        h.Color =  [1 1 1];
        errorshaded(x, mu_y, sqrt(sigma2_y), 'Color', c(22,:),'DisplayName','Prediction', 'Opacity', 1); hold on;
        plot(x,mu_y,'LineWidth',1.5,'Color', 'k'); hold on;
        plot(x,y,'LineWidth',1.5,'Color', c(end,:)); hold on;
        scatter(x_tr, y_tr, 'MarkerFaceColor', c(end,:), 'MarkerEdgeColor', 'none'); hold on;
        scatter(new_x, new_y, '*','MarkerFaceColor',c(end,:),  'MarkerEdgeColor', c(end,:), 'LineWidth',1.5) ; hold off;
        xlabel('x','Fontsize',Fontsize)
        ylabel('f(x)','Fontsize',Fontsize)
        set(gca, 'Fontsize', Fontsize, 'Xlim', Xlim,  'Ylim',Ylim)
        grid off
        box off
        F(i) = getframe(gcf);
        
    else
        i_tr= idx(i); %random sampling
        new_x = x(:,i_tr);
        new_y = y(:,i_tr); % no noise %%%%%%%%%%%%%%%%%%%
    end
    
    h=figure(5);
    h.Name = 'Cumulative regret';
    h.Color =  [1 1 1];
    plot(0:maxiter,cum_regret,'LineWidth',1.5,'Color', c(end,:)); hold off;
    xlabel('Iteration','Fontsize',Fontsize)
    ylabel('Cumulative regret','Fontsize',Fontsize)
    set(gca, 'Fontsize', Fontsize, 'Xlim', [0, maxiter])
    grid off
    box off

    if i == 5
        h=figure(2);
        h.Name = 'Bayesian optimisation before hyperparameters optimization';
        h.Color =  [1 1 1];
        subplot(2,1,1)
        errorshaded(x, mu_y, sqrt(sigma2_y), 'Color', cmap(1,:),'DisplayName','Prediction', 'Opacity',0.5); hold on
        p1 = plot(x,mu_y,'LineWidth',linewidth, 'Color', cmap(1,:)); hold on;
        p2 = scatter(x_tr, y_tr,markersize, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'none'); hold on;
        p3 = plot(x,y,'LineWidth',linewidth, 'color', cmap(end,:)); hold off;
        set(gca,'XTick',[], 'YTick', [])
        %         xlabel('Encoder parameters','Fontsize',Fontsize)
        ylabel('Percept quality','Fontsize',Fontsize)
        set(gca, 'Fontsize', Fontsize, 'Xlim', Xlim)
        grid off
        box off
        legend([p1, p3], 'Inferred objective','True objective') 
        legend boxoff

        subplot(2,1,2)
        plot(x,EI,'LineWidth',linewidth,'Color', cmap(end,:)); hold on;
        xlabel('Encoder parameters','Fontsize',Fontsize)
        ylabel('Sample utility','Fontsize',Fontsize)
        set(gca,'XTick',[], 'YTick', [])
        box off
        vline(new_x,'Linewidth',linewidth, 'ymax', max_EI); hold off;
        text(new_x,max_EI+0.03,'Next sample')
        exportgraphics(h,'GP_BO.png')
    end
    
end

fig = figure;
fps =1;
movie(fig,F,1, fps)

% create the video writer with 1 fps

writerObj = VideoWriter('BO.avi');
writerObj.FrameRate = fps;
% set the seconds per image
% open the video writer
open(writerObj);
% write the frames to the video
for i=1:length(F)
    % convert the image to a frame
    frame = F(i) ;
    writeVideo(writerObj, frame);
end
% close the writer object
close(writerObj);

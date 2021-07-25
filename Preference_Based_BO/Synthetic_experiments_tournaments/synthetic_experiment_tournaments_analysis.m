acquisition_fun_list = {'P_GP_UCB','decorrelatedsparring_grid','kernelselfsparring_grid', 'random', 'new_DTS_grid'};
nacq_func = numel(acquisition_fun_list);
objective = 'GP1d'; %chose an objective function: 'forretal08'
for a = 1:nacq_func
    results=  load(['../Data/synthetic_experiments_tournaments_data_fixed_theta/grid', objective, '_', acquisition_fun_list{a}], 'experiment');
    results = results.experiment;
    
    if a==1
        maxiter = results.maxiter;
        nreplicates= results.nreplicates;
        gxCa = NaN(nacq_func, maxiter);
        gxga = NaN(nacq_func, maxiter);
        gxca = NaN(nacq_func, maxiter);
        gmaxr = NaN(nacq_func, nreplicates);
    end
    
    gxc= NaN(results.nreplicates, results.maxiter);
    gxg= NaN(results.nreplicates, results.maxiter);
    
    for r =1:results.nreplicates
        n=['r',num2str(r)];
        rr= results.(n);
        gxg(r,:) = rr.grange(rr.idxmaxg);
        gxc(r,:) = rr.grange(rr.idxmaxc);
        gmaxr(a,r)= rr.gmax;
    end
    gxga(a,:) = mean(gxg,1);
    gxca(a,:) = mean(gxc,1);
end
gmax = mean(gmaxr(1,:));

acquisition_fun_list = {'DecorrelatedSelfSparring','KernelSelfSparring'};

Fontsize =14;
%% Plot the estimate of gmax over time
% Using mean g
fig=figure(2);
fig.Color =  [1 1 1];
fig.Name = 'Estimated maximum based on the predictive mean of the value function';
plot(1:maxiter, gxga,'LineWidth',1.5); hold on;
plot(1:maxiter, gmax*ones(1,maxiter), 'Color', 'k','LineWidth',1.5); hold off;
xlabel('t','Interpreter','latex','Fontsize',Fontsize)
ylabel('$g*$','Interpreter','latex','Fontsize',Fontsize)
%title('1D Gaussian processes','Fontsize',Fontsize, 'FontName',Fontname)
title('1D Gaussian processes','Interpreter','latex','Fontsize',Fontsize)
legend([acquisition_fun_list, 'Maximum'], 'Location', 'southeast','Interpreter','latex','Fontsize',Fontsize)
% %Using mean mu_c
% fig=figure(3);
% fig.Color =  [1 1 1];
% fig.Name = 'Estimated maximum based on the maximum of Phi(f(x,x''))';
% plot(1:maxiter, gxca,'LineWidth',1.5); hold on;
% plot(1:maxiter, gmax*ones(1,maxiter), 'Color', 'k','LineWidth',1.5); hold off;
% xlabel('t','Fontsize',Fontsize)
% ylabel('g*','Fontsize',Fontsize)
% box off
% legend([acquisition_fun_list, 'Maximum'], 'Location', 'southeast')


%% Compute g* by sampling the distribution p(x*|D), and then g* = g(x*)
kernelfun = @Gaussian_kernelfun_wnoise;
modeltype = 'exp_prop'; % Approximation model
gxmax = zeros(nacq_func,maxiter, nreplicates);
idmax = zeros(nacq_func,maxiter, nreplicates);
nsamples=100;
for a = 1:nacq_func
    results=  load(['./synthetic_experiments_data/grid', objective, '_', acquisition_fun_list{a}], 'experiment');
    results = results.experiment;    
    theta = [6 ; 1; -15];
    for i = 1:10:results.maxiter        
        for r =1:results.nreplicates
            n=['r',num2str(r)];
            rr= results.(n);
            xtrain = rr.xtrain(:,1:i);
            ctrain = rr.ctrain(:,1:i);
            [mu_c,  mu_y, sigma2_y, Sigma2_y] = prediction_bin(theta, xtrain, ctrain, xtrain, kernelfun, modeltype, post, regularization);
            for k=1:nsamples
                fsamples = mvnrnd(mu_y,  Sigma2_y)'; %sample from the posterior at training points
                sample_g = sample_value_GP(rr.x, theta, xtrain, fsamples, Sigma2_y);
                [gmax, idmax(a,i,r)]=max(sample_g);
                gxmax(a,i,r) = gxmax(a,i,r) + rr.grange(idmax(a,i,r));
            end
        end
    end
end
%get the mode of p(x*|D)
gxmax(a,i,r) = gxmax(a,i,r)/nsamples;
[f,xi] = ksdensity(x);


test = mean(gxmax,3);
test= test(:,1:10:maxiter);
figure()
plot(1:10:maxiter, test')

% Using mean g
fig=figure(2);
fig.Color =  [1 1 1];
fig.Name = 'Distance between the estimated maximum Estimated maximum based on the predictive mean of the value function';
plot(1:10:maxiter, test','LineWidth',1.5); hold on;
plot(1:maxiter, gmax*ones(1,maxiter), 'Color', 'k','LineWidth',1.5); hold off;
xlabel('t','Fontsize',Fontsize)
ylabel('g*','Fontsize',Fontsize)
box off
legend([acquisition_fun_list, 'Maximum'])
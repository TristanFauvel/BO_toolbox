
acquisition_fun_list = {'P_GP_UCB', 'decorrelatedsparring_grid','kernelselfsparring_grid', 'random', 'new_DTS_grid'};
nacq_func = numel(acquisition_fun_list);
objective = 'GP1d'; %chose an objective function: 'forretal08'
nd= 1;
for a = 1:nacq_func
    for m = 1:nd
        results=  load(['../Data/synthetic_experiments_tournaments_data_fixed_theta/grid', objective, '_', acquisition_fun_list{a}, '_m',num2str(m+1)], 'experiment');
        results = results.experiment;
        
        if a==1 && m==1
            maxiter = results.maxiter;
            nreplicates= results.nreplicates;
            gxCa = NaN(nacq_func,nd-1, maxiter);
            gxga = NaN(nacq_func,nd-1, maxiter);
            gxca = NaN(nacq_func,nd-1, maxiter);
            gx = NaN(nacq_func,nd-1, nreplicates, maxiter);
            gmaxr = NaN(nacq_func,nd-1, nreplicates);
        end
        
        gxc= NaN(nreplicates, results.maxiter);
        gxg= NaN(nreplicates, results.maxiter);
        
        for r =1:nreplicates
            n=['r',num2str(r)];
            rr= results.(n);
            gxg(r,:) = rr.grange(rr.idxmaxg);
            gxc(r,:) = rr.grange(rr.idxmaxc);
            gmaxr(a,m,r)= rr.gmax;
            gx(a,m,r,:)=rr.grange(rr.idxmaxg);
        end
        gxga(a,m,:) = mean(gxg,1);
        gxca(a,m,:) = mean(gxc,1);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gmax = mean(gmaxr(1,1,:));

names = {'DecorrelatedSelfSparring', 'KernelSelfSparring', 'random','DTS'};
k=0;
for a = 1:nacq_func
    for m=1:nd
        k=k+1;
        legends{k}=[names{a}, ', m=',num2str(m+1)];
    end
end

%legends = {'DTS', 'DecorrelatedSelfSparring','KernelSelfSparring', 'random'};

set(0,'defaulttextInterpreter','latex') 

Fontsize =14;
fig=figure();
fig.Color =  [1 1 1];
fig.Name = 'Estimated maximum based on the predictive mean of the value function';
for a = 1:nacq_func
    for m = 1:nd
        plot(1:maxiter, squeeze(gxga(a,m,:)),'LineWidth',1.5); hold on;
    end
end
plot(1:maxiter, gmax*ones(1,maxiter), 'Color', 'k','LineWidth',1.5); hold on;
hold off;
xlabel('t','Interpreter','latex','Fontsize',Fontsize)
ylabel('$g*$','Interpreter','latex','Fontsize',Fontsize)
title('1D Gaussian processes','Interpreter','latex','Fontsize',Fontsize)
legend([legends, 'Maximum'], 'Location', 'southeast','Interpreter','latex','Fontsize',Fontsize)


for m = 1:nd
    fig=figure();
    fig.Color =  [1 1 1];
    fig.Name = 'Estimated maximum based on the predictive mean of the value function';  
    for a = 1:nacq_func        
        plot(1:maxiter, squeeze(gxga(a,m,:)),'LineWidth',1.5); hold on;
    end
    plot(1:maxiter, gmax*ones(1,maxiter), 'Color', 'k','LineWidth',1.5); hold on;        
    hold off;
    xlabel('t','Interpreter','latex','Fontsize',Fontsize)
    ylabel('$g*$','Interpreter','latex','Fontsize',Fontsize)
    title(['1D Gaussian processes, m=', num2str(m+1)],'Fontsize',Fontsize)
    legend([legends, 'Maximum'], 'Location', 'southeast','Interpreter','latex','Fontsize',Fontsize)
end

for a = 1:nacq_func
    fig=figure();
    fig.Color =  [1 1 1];
    fig.Name = 'Comparison m=1 and m=2';
    color= (1:maxiter)/255;
    scatter(gxga(a,1,:),gxga(a,4,:), 15,color, 'filled'); hold on;
    plot(linspace(min(gxga(a,1,:)), max(gxga(a,1,:))),linspace(min(gxga(a,1,:)), max(gxga(a,1,:))),'Color', 'k','LineWidth',1.5);
    title(['1D Gaussian processes', legends(a)],'Fontsize',Fontsize)
    xlabel('$g*, m=2$','Fontsize',Fontsize)
    ylabel('$g*, m=5$','Fontsize',Fontsize)
end



for a = 1:nacq_func
    fig=figure();
    fig.Color =  [1 1 1];
    fig.Name = 'Comparison m=1 and m=2';
    color= (1:maxiter)/255;
    for m = 1:(nd-1)
        scatter(m*ones(1,maxiter),gxga(a,m,:), 15,color, 'filled'); hold on;
    end
    hold off;
    title(['1D Gaussian processes', legends(a)],'Fontsize',Fontsize)
    xlabel('$g*$','Fontsize',Fontsize)
    ylabel('$m$','Fontsize',Fontsize)
end


% gx = NaN(nacq_func,nd-1, nreplicates, maxiter);
for a = 1:nacq_func
    fig=figure();
    fig.Color =  [1 1 1];
    fig.Name = 'Comparison m=1 and m=2';
    color= (1:maxiter)/255;
    color=repmat(color,1,nreplicates);
    scatter(reshape(gx(a,1,:,:),[],1),reshape(gx(a,2,:),[],1), 15,color, 'filled'); hold on;
    plot(linspace(min(gxga(a,1,:)), max(gxga(a,1,:))),linspace(min(gxga(a,1,:)), max(gxga(a,1,:))),'Color', 'k','LineWidth',1.5);
    title(['1D Gaussian processes', legends(a)],'Fontsize',Fontsize)
    xlabel('$g*, m=2$','Fontsize',Fontsize)
    ylabel('$g*, m=2$','Fontsize',Fontsize)
end


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
legend([legends, 'Maximum'], 'Location', 'southeast','Interpreter','latex','Fontsize',Fontsize)

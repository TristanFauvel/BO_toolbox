%Preference learning: synthetic experiments

% function preference_learning(randinit, varargin)
clear all
upath = userpath;
addpath(genpath([upath,'/GP_toolbox']))
addpath(genpath([upath,'/BO_toolbox']))
addpath(genpath([upath,'/Preference_Based_BO_toolbox']))

addpath(genpath('/home/tfauvel/Desktop/optim_retina'))
cd([upath,'/Preference_Based_BO_toolbox/Code'])

close all


acquisition_fun_list = {'PF_acq', 'random', 'kernelselfsparring'};
% acquisition_fun_list = {'kernelselfsparring_tournament_grid'};
% acquisition_fun_list = {'P_GP_UCB_grid'};
seed = 124;

nacq_func = numel(acquisition_fun_list);


%% First experiment : generate gaussian processes
% optim=NaN(nacq_func, nreplicates, maxiter);
nreplicates =1; %20;
maxiter = 50; %total number of iterations : 200

m=2;
for m =2:2  %tournament size
    for a = 1:nacq_func
        seed=130; %seed+1;
        for r=1:nreplicates
            seed=seed+1;
            acquisition_fun = str2func(acquisition_fun_list{a});
            objective= 'GP1d';
            results =  tournament_learning_synthetic(acquisition_fun, objective, seed, maxiter, ntest, m);
            results.seed(r) = seed;
            n=['r',num2str(r)];
            experiment.(n) = results;
        end
        experiment.nreplicates = nreplicates;
        experiment.acquisition_fun = acquisition_fun;
        experiment.objective = objective;
        experiment.ntest = ntest;
        experiment.maxiter= maxiter;
        experiment.m = m;
        save(['../Data/synthetic_experiments_tournaments_data_fixed_theta/', objective, '_', acquisition_fun_list{a},'_m',num2str(m)], 'experiment')
    end
end

%% Second experiment : generate 2d GPgaussian processes
% optim=NaN(nacq_func, nreplicates, maxiter);
% nreplicates = 20;
% maxiter = 200; %total number of iterations : 200

% for m =2:6  %tournament size
    for a = 1:nacq_func
        seed=130; %seed+1;
        for r=1:nreplicates
            seed=seed+1;
            acquisition_fun = str2func(acquisition_fun_list{a});
            objective= 'GP2d';
            results =  tournament_learning_synthetic(acquisition_fun, objective, seed, maxiter, ntest, m);
            results.seed(r) = seed;
            n=['r',num2str(r)];
            experiment.(n) = results;
        end
        experiment.nreplicates = nreplicates;
        experiment.acquisition_fun = acquisition_fun;
        experiment.objective = objective;
        experiment.ntest = ntest;
        experiment.maxiter= maxiter;
        save(['../Data/synthetic_experiments_tournaments_data_fixed_theta/', objective, '_', acquisition_fun_list{a},'_m',num2str(m)], 'experiment')
    end
% end

%same thing in ten dimensions
m=2;
acquisition_fun_list = {'kernelselfsparring_tournament','new_DTS_tournament', 'random'};
for a = 1:nacq_func
    seed=130; %seed+1;
    for r=1:nreplicates
        seed=seed+1;
        acquisition_fun = str2func(acquisition_fun_list{a});
        objective= 'GP10d';
        results =  tournament_learning_synthetic(acquisition_fun, objective, seed, maxiter, ntest, m);
        results.seed(r) = seed;
        n=['r',num2str(r)];
        experiment.(n) = results;
    end
    experiment.nreplicates = nreplicates;
    experiment.acquisition_fun = acquisition_fun;
    experiment.objective = objective;
    experiment.ntest = ntest;
    experiment.maxiter= maxiter;
    experiment.m = m;
    save(['../Data/synthetic_experiments_tournaments_data_fixed_theta/', objective, '_', acquisition_fun_list{a},'_m',num2str(m)], 'experiment')
end
        
%% Third experiment : benchmarks 1d
ntest= 1000; %33
nreplicates = 1; %40;
maxiter = 50; %total number of iterations : 200

for a = 1:nacq_func
    seed=130; %seed+1;
    for r=1:nreplicates
        seed=seed+1;
        acquisition_fun = str2func(acquisition_fun_list{a});
        objective= 'forretal08';
        results =  tournament_learning_synthetic(acquisition_fun, objective, seed, maxiter, ntest);
        results.seed(r) = seed;
        n=['r',num2str(r)];
        experiment.(n) = results;
    end
    experiment.nreplicates = nreplicates;
    experiment.acquisition_fun = acquisition_fun;
    experiment.objective = objective;
    experiment.ntest = ntest;
    experiment.maxiter= maxiter;
    save(['../Data/synthetic_experiments_tournaments_data_fixed_theta/', objective, '_', acquisition_fun_list{a}], 'experiment')
end

%% Fourth experiment : benchmarks 2d
objective_list = {'levy',  'goldpr', 'camel6'}; %chose an objective function: 'forretal08'
ntest= 100; %33
nreplicates = 1;%20;
maxiter = 50; %total number of iterations : 200
for i = 1:numel(objective_list)
    objective = objective_list{i};
    for a = 1:nacq_func
        seed=130; %seed+1;
        for r=1:nreplicates
            seed=seed+1;
            acquisition_fun = str2func(acquisition_fun_list{a});
            results =  tournament_learning_synthetic(acquisition_fun, objective, seed, maxiter, ntest);
            results.seed(r) = seed;
            n=['r',num2str(r)];
            experiment.(n) = results;
        end
        experiment.nreplicates = nreplicates;
        experiment.acquisition_fun = acquisition_fun;
        experiment.objective = objective;
        experiment.ntest = ntest;
        experiment.maxiter= maxiter;
        save(['../Data/synthetic_experiments_tournaments_data_fixed_theta/', objective, '_', acquisition_fun_list{a}], 'experiment')
    end
end

% toplot= squeeze(mean(optim, 2))';
% 
% experiment.results = optim;
% experiment.nreplicates = nreplicates;
% experiment.acquisition_fun_list=  acquisition_fun_list;
% experiment.ntest = ntest;
% experiment.maxiter= maxiter;
% experiment.answer = answer;
% experiment.seed = seed;
% save('synthetic_preference_experiment_tournaments_MES.mat', 'experiment')
% 
% figure()
% plot(toplot)
% legend(acquisition_fun_list(1:3))
% 
% std_g = NaN(nacq_func, maxiter);
% for i = 1:nacq_func
%     std_g(i,:)= sqrt(var(squeeze(optim(i,:,:))));
% end


% cmap = gray(256);
% colo= othercolor('GnBu7');
% colo = jet;
% Fontsize = 14;
% fig=figure(5);
% fig.Name = 'Estimated maximum';
% fig.Color =  [1 1 1];
% for i = 1:nacq_func
%     errorshaded(1:maxiter,toplot(i,:), std_g(i,:), 'Color',  colo(floor(64*i/nacq_func),:),'LineWidth', 1.5, 'Fontsize', 14); hold on
% end
% hold off;
% xlabel('Iteration','Fontsize',Fontsize)
% ylabel('g(x_c)', 'Fontsize',Fontsize)
% box off;


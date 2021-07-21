%Preference learning: synthetic experiments

% function preference_learning(randinit, varargin)
clear all 
addpath(genpath('././GP_toolbox'))
addpath(genpath('././BO_toolbox'))
addpath(genpath('././optim_retina/psychophysics'))
addpath(genpath('./experiment_funs'))
addpath(genpath('./my_tools'))

close all


acquisition_fun_list = {'MES', 'brochu_EI', 'bivariate_EI', 'duel_thompson', 'PES', 'new_DTS',  'copeland_expected_improvement', 'random', 'value_expected_improvement', 'active_sampling'};
acquisition_fun_list = {'decorrelatedsparring','kernelselfsparring', 'brochu_EI', 'bivariate_EI', 'duel_thompson', 'new_DTS', 'random','active_sampling','MES','value_expected_improvement', 'copeland_expected_improvement'};
acquisition_fun_list = {'MES','value_expected_improvement', 'copeland_expected_improvement'};

acquisition_fun_list = {'UCB_LWB', 'kernelselfsparring', 'new_DTS'};
acquisition_fun_list = {'PF_acq_grid', 'random',  'kernelselfsparring'};

objective = 'forretal08'; %chose an objective function: 'forretal08'

seed = 124;

%there is a problem with 'value_expected_improvement'
maxiter = 200; %total number of iterations : 200

%answer = 'max_mu_g'; %the way I report the maximum, either by maximizing the predictive mean of g, either by maximizing the soft-copeland score, EITHER BY maximizing MU_C: which makes more sense (but this depends on the acquisition function).
%answer = 'Copeland_score'; %the way I report the maximum, either by maximizing the predictive mean of g, either by maximizing the soft-copeland score.
%answer = 'max_g'

ntest= 33; %33

nreplicates = 20;

nacq_func = numel(acquisition_fun_list);


%% First experiment : generate gaussian processes
% optim=NaN(nacq_func, nreplicates, maxiter);
for a = 1:nacq_func
    seed=130; %seed+1;
    for r=1:nreplicates
        seed=seed+1;
        acquisition_fun = str2func(acquisition_fun_list{a});
        objective= 'GP';
        results =  preference_learning_synthetic_on_grid(acquisition_fun, objective, seed, maxiter, ntest);
        results.seed(r) = seed;
        n=['r',num2str(r)];
        experiment.(n) = results;
    end
    experiment.nreplicates = nreplicates;
    experiment.acquisition_fun = acquisition_fun;
    experiment.objective = objective;
    experiment.ntest = ntest;
    experiment.maxiter= maxiter;
    save(['./synthetic_experiments_data/grid', objective, '_', acquisition_fun_list{a},'_33'], 'experiment')
end

%% First experiment : benchmarks
ntest= 33; %33


for a = 1:nacq_func
    seed=130; %seed+1;
    for r=1:nreplicates
        seed=seed+1;
        acquisition_fun = str2func(acquisition_fun_list{a});
        objective= 'forretal08';
        results =  preference_learning_synthetic_on_grid(acquisition_fun, objective, seed, maxiter, ntest);
        results.seed(r) = seed;
        n=['r',num2str(r)];
        experiment.(n) = results;
    end
    experiment.nreplicates = nreplicates;
    experiment.acquisition_fun = acquisition_fun;
    experiment.objective = objective;
    experiment.ntest = ntest;
    experiment.maxiter= maxiter;
    save(['./synthetic_experiments_data/grid', objective, '_', acquisition_fun_list{a}], 'experiment')
end

ntest= 500; %33
for a = 1:nacq_func
    seed=130; %seed+1;
    for r=1:nreplicates
        seed=seed+1;
        acquisition_fun = str2func(acquisition_fun_list{a});
        objective= 'GP';
        results =  preference_learning_synthetic_on_grid(acquisition_fun, objective, seed, maxiter, ntest);
        results.seed(r) = seed;
        n=['r',num2str(r)];
        experiment.(n) = results;
    end
    experiment.nreplicates = nreplicates;
    experiment.acquisition_fun = acquisition_fun;
    experiment.objective = objective;
    experiment.ntest = ntest;
    experiment.maxiter= maxiter;
    save(['./synthetic_experiments_data/grid', objective, '_', acquisition_fun_list{a}], 'experiment','_33')
end

toplot= squeeze(mean(optim, 2))';

experiment.results = optim;
experiment.nreplicates = nreplicates;
experiment.acquisition_fun_list=  acquisition_fun_list;
experiment.ntest = ntest;
experiment.maxiter= maxiter;
experiment.answer = answer;
experiment.seed = seed;
save('synthetic_preference_experiment_MES.mat', 'experiment')

figure()
plot(toplot)
legend(acquisition_fun_list(1:3))

std_g = NaN(nacq_func, maxiter);
for i = 1:nacq_func
    std_g(i,:)= sqrt(var(squeeze(optim(i,:,:))));
end


cmap = gray(256);
colo= othercolor('GnBu7');
colo = jet;
Fontsize = 14;
fig=figure(5);
fig.Name = 'Estimated maximum';
fig.Color =  [1 1 1];
for i = 1:nacq_func
    errorshaded(1:maxiter,toplot(i,:), std_g(i,:), 'Color',  colo(floor(64*i/nacq_func),:),'LineWidth', 1.5, 'Fontsize', 14); hold on
end
hold off;
xlabel('Iteration','Fontsize',Fontsize)
ylabel('g(x_c)', 'Fontsize',Fontsize)
box off;

%%
figure()
plot(squeeze(mean(experiment.results,2))')
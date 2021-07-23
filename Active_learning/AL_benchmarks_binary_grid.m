clear all
close all

add_bo_module;

seed=1;
maxiter= 30;% 100;
rng(seed)


options_theta.method = 'lbfgs';
options_theta.verbose = 1;

update_period = maxiter+ 2;

ninit = maxiter +2 ;
nopt= 5;% number of time steps before starting using the acquisition functions.
nrepets = 5; %20;
seeds=1:nrepets;


nobj = 6;
objectives = cell(1,nobj);
for i = 1:nobj
    objectives{i} = 'japan';
end
kernelnames = {'Matern32', 'Matern32','Matern52','Matern52','Gaussian','Gaussian'};
lengthscales = {'long', 'short','long', 'short','long', 'short'};


acquisition_funs = {'TME_sampling_binary', 'maxvar_binary_grid', 'random','BALD_grid'};
% acquisition_funs = {'random','BALD_grid'};


nobj = numel(objectives);
nacq = numel(acquisition_funs);
for j = 1:nobj
    bias = 0;
    objective = [objectives{j}, '_', kernelnames{j}, '_',lengthscales{j}];
    [x, y, theta, lb, ub, theta_lb, theta_ub, kernelfun] = load_benchmarks_active_learning_grid(objectives{j}, kernelnames{j}, lengthscales{j});
    for ji = 1:nacq
        acquisition_name = acquisition_funs{ji};
        acquisition_fun = str2func(acquisition_name);
        clear('xtrain', 'ctrain', 'cum_regret', 'score');
        for k = 1:nrepets
            disp(['Repetition ', num2str(k)])
            seed = seeds(k);
            [xtrain{k}, ctrain{k}, cum_regret{k}, score{k}]= AL_loop_binary_grid(x, ...
                y, maxiter, nopt, kernelfun, theta, acquisition_fun, ninit, theta_lb, theta_ub, lb, ub, seed);            
        end
        clear('experiment')
        fi = ['xtrain_',acquisition_name];
        experiment.(fi) = xtrain;
        fi = ['xtrain_norm_',acquisition_name];
        experiment.(fi) = ctrain;
        fi = ['score_',acquisition_name];
        experiment.(fi) = score;
        fi = ['cul_regret_',acquisition_name];
        experiment.(fi) = score;
        filename = [pathname,'/Active_learning/Data_active_learning_binary_grid/',objective,'_',acquisition_name];
        save(filename, 'experiment')
    end
end
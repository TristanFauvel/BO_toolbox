clear all

add_bo_module;
close all

data_dir =  [pathname,'/Binary_BO/Data/'];


acquisition_funs = {'EI_Tesch', 'TS_binary','random_acquisition_binary', 'UCB_binary', 'UCB_binary_latent','BKG'}; %, 'bivariate_EI_binary'};
% acquisition_funs = {'random_acquisition_binary'} %, 'UCB_binary_latent', 'bivariate_EI_binary'};
 
 maxiter = 50; %total number of iterations : 200

nreplicates = 20;

nacq = numel(acquisition_funs);
nbo = 2;

seeds = 1:nreplicates;
update_period = maxiter+2; % do not update the hyperparameters;

rescaling = 1;
if rescaling == 0
    load('benchmarks_table.mat')
else
    load('benchmarks_table_rescaled.mat')
end
load('benchmarks_table.mat') %%%%%%%%%%%%%%%%%%%%%%


objectives = benchmarks_table.fName;  
nobj =numel(objectives);


link = @normcdf;
 for j = 1:nobj %%%%%%%%%%%%%%%%
    objective = char(objectives(j));
    [g, theta, model] = load_benchmarks(objective, [], benchmarks_table, rescaling);
    close all
    model.ns = 0;
    model.link = link;
    model.type = 'classification';
    model.task = 'max';
    for a =1:nacq
        acquisition_name = acquisition_funs{a};
        if strcmp(acquisition_name, 'BKG')
            modeltype = 'laplace';
        else
            modeltype = 'exp_prop';
        end
        model.modeltype = modeltype;
        
        acquisition_fun = str2func(acquisition_name);
        clear('xtrain', 'xtrain_norm', 'ctrain', 'score');
        for r=1:nreplicates
            seed  = seeds(r)
            [xtrain{r}, xtrain_norm{r}, ctrain{r}, score{r}, xbest{r}] =  BBO_loop(acquisition_fun, nbo, seed, maxiter, theta, g, update_period, model);
        end
        save_benchmark_results(acquisition_name, xtrain, xtrain_norm, ctrain, score, xbest, g, objective, data_dir)
        
    end
end

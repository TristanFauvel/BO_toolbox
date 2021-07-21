clear all

add_bo_module;
close all

data_dir =  [pathname,'/Binary_BO/Data/'];


acquisition_funs = {'TS_binary', 'random_acquisition_binary', 'KG_binary'};

maxiter = 60; %total number of iterations : 200

nreplicates = 40; %20;

nacq = numel(acquisition_funs);
nbo = 2;

objectives = {'forretal08', 'grlee12', 'GP1d','levy', 'goldpr', 'camel6','Ursem_waves'};
% objectives = {'levy', 'goldpr', 'camel6','Ursem_waves'};

nobj =numel(objectives);
seeds = 1:nreplicates;
update_period = maxiter+2; % do not update the hyperparameters;
for j = 1:nobj
    objective = objectives{j};
    
    kernelfun = @ARD_kernelfun;
    kernelname = 'ARD';
    link = @normcdf;
    modeltype = 'exp_prop';
    [g, theta, lb, ub, lb_norm, ub_norm, theta_lb, theta_ub] = load_benchmarks(objective, kernelname);
    close all
    for a =1:1%:nacq
        acquisition_name = acquisition_funs{a};
        acquisition_fun = str2func(acquisition_name);
        clear('xtrain', 'xtrain_norm', 'ctrain', 'score');
        for r=1:nreplicates
            seed  = seeds(r)
            [xtrain{r}, xtrain_norm{r}, ctrain{r}, score{r}] =  BBO_loop(acquisition_fun, nbo, seed, lb, ub, maxiter, theta, g, update_period, modeltype, theta_lb, theta_ub, kernelname, kernelfun, lb_norm, ub_norm, link);
        end
        clear('experiment')
        fi = ['xtrain_',acquisition_name];
        experiment.(fi) = xtrain;
        fi = ['xtrain_norm_',acquisition_name];
        experiment.(fi) = xtrain_norm;
        fi = ['ctrain_',acquisition_name];
        experiment.(fi) = ctrain;
        fi = ['score_',acquisition_name];
        experiment.(fi) = score;
        
        filename = [data_dir,objective,'_',acquisition_name];
        close all
        save(filename, 'experiment')
    end
end

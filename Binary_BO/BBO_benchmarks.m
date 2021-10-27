clear all

add_bo_module;

data_dir =  [pathname,'/Binary_BO/Data/'];

 
acquisition_funs = {'EI_Tesch', 'TS_binary','random_acquisition', 'UCB_binary', 'UCB_binary_latent'}; %, 'bivariate_EI_binary'};

maxiter = 100;
nreplicates = 60;
update_period = 15000;

ninit = 5000;

nacq = numel(acquisition_funs);
nopt = 2;

seeds = 1:nreplicates;
update_period = maxiter+2; % do not update the hyperparameters;

rescaling = 1;
if rescaling == 0
    load('benchmarks_table.mat')
else
    load('benchmarks_table_rescaled.mat')
end

objectives = benchmarks_table.fName;
nobj =numel(objectives);
task = 'max';
hyps_update = 'none';
link = @normcdf;
identification = 'mu_c';
ns = 0;

for j = 1:nobj
    objective = char(objectives(j));
    [g, theta, model] = load_benchmarks(objective, [], benchmarks_table, rescaling, 'classification');
    close all    
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

        optim = binary_BO(g, task, identification, maxiter, nopt, ninit, update_period, hyps_update, acquisition_fun, ns);

        for k=1:nreplicates
            seed  = seeds(k)
            [xtrain{k}, xtrain_norm{k}, ctrain{k}, score{k}, xbest{k}]= optim.optimization_loop(seed, theta, model);
        end
        fi = ['xtrain_',acquisition_name];
        experiment.(fi) = xtrain;
        fi = ['xtrain_norm_',acquisition_name];
        experiment.(fi) = xtrain_norm;
        fi = ['ctrain_',acquisition_name];
        experiment.(fi) = ctrain;

        fi = ['score_',acquisition_name];
        experiment.(fi) = score;
        fi = ['xbest_',acquisition_name];
        experiment.(fi) = xbest;

        filename = [data_dir,objective,'_',acquisition_name];
        %         save(filename, 'experiment')        

    end
end


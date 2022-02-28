function BBO_benchmarks(pathname)

data_dir =  [pathname,'/Data/Data_BBO'];

 
 
settings= load([pathname, '/Experiments_parameters.mat'],'Experiments_parameters');
settings = settings.Experiments_parameters;
settings = settings({'BBO'},:);

% List of acquisition functions tested in the experiment
acquisition_funs = settings.acquisition_funs{:};
maxiter = settings.maxiter;
nreplicates = settings.nreplicates;
ninit = settings.ninit;
nopt = settings.nopt;
nacq = numel(acquisition_funs);
task =  settings.task{:};
hyps_update = settings.hyps_update{:};
link = settings.link{:};
identification = settings.identification{:};
ns = settings.ns;
update_period = settings.update_period;
modeltype = settings.modeltype;
rescaling = settings.rescaling;
seeds = 1:nreplicates;
more_repets = 0;

if rescaling == 0
    load('benchmarks_table.mat')
else
    load('benchmarks_table_rescaled.mat')
end

objectives = settings.objectives{:};
nobj =numel(objectives);

for j = 1:nobj
    objective = char(objectives(j));
    close all    
      [g, theta, model] = load_benchmarks(objective, [], benchmarks_table, rescaling, 'classification', 'modeltype', modeltype, 'link', link);
  
    for a =1:nacq
        acquisition_name = acquisition_funs{a};      
        acquisition_fun = str2func(acquisition_name);
        clear('xtrain', 'xtrain_norm', 'ctrain', 'score');

        optim = binary_BO(g, task, identification, maxiter, nopt, ninit, update_period, hyps_update, acquisition_fun, model.D, ns);


        if more_repets == 1
            load(filename, 'experiment')
            n = numel(experiment.(['xtrain_',acquisition_name]));
            for k = 1:nrepets
                
                disp(['Repetition : ', num2str(n+k)])
                seed =n+k;
                [experiment.(['xtrain_',acquisition_name]){n+k}, experiment.(['xtrain_norm_',acquisition_name]){n+k}, experiment.(['ctrain_',acquisition_name]){n+k}, experiment.(['score_',acquisition_name]){n+k},experiment.(['xbest_',acquisition_name]){n+k}]= optim.optimization_loop(seed, theta, model);
            end
            save(filename, 'experiment')            
        else
            for r=1:nreplicates   
                seed  = seeds(r);
                [xtrain{r}, ctrain{r}, score{r}, xbest{r}] = optim.optimization_loop(seed, theta, model);
            end
            structure_name= acquisition_name;
            save_benchmark_results(acquisition_name, structure_name, xtrain, ctrain, score, xbest, objective, data_dir, task)
        end
    end
end

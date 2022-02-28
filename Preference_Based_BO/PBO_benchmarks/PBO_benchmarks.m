function PBO_benchmarks(pathname)


%Preference learning: synthetic experiments

data_dir =  [pathname,'/Data/Data_PBO'];

 
settings= load([pathname, '/Experiments_parameters.mat'],'Experiments_parameters');
settings = settings.Experiments_parameters;
settings = settings({'PBO'},:);

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
if rescaling ==0
    load('benchmarks_table.mat')
else
    load('benchmarks_table_rescaled.mat')
end

objectives = settings.objectives{:};
nobj =numel(objectives);

seeds = 1:nreplicates;
more_repets = 0;

for j = 1:nobj
    objective = char(objectives(j));     
            [g, theta, model] = load_benchmarks(objective, [], benchmarks_table, rescaling, 'preference', 'modeltype', modeltype, 'link',link);
    for a =1:nacq
         acquisition_name = acquisition_funs{a};
%         if strcmp(acquisition_name, 'PKG')
%             modeltype = 'laplace';
%         else
%             modeltype = 'exp_prop';
%         end
 
        acquisition_fun = str2func(acquisition_name);
        clear('xtrain', 'xtrain_norm', 'ctrain', 'score');
        
        filename = [data_dir,objective,'_',acquisition_name];
        
        optim = preferential_BO(g, task, identification, maxiter, nopt, ninit, update_period, hyps_update, acquisition_fun, model.D, ns);

        if more_repets==1
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

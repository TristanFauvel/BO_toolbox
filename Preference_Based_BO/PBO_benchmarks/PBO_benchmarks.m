%Preference learning: synthetic experiments
add_bo_module;
close all


acquisition_funs = {'bivariate_EI','Dueling_UCB','EIIG','random_acquisition_pref','kernelselfsparring','MUC','Brochu_EI', 'Thompson_challenge','DTS'};

acquisition_funs = {'Thompson_challenge','DTS'};
maxiter = 80;
nreplicates = 40;

maxiter = 20;
nreplicates = 1; 

ninit = 5;
nopt = 5;
nacq = numel(acquisition_funs);

rescaling = 1;
if rescaling ==0
    load('benchmarks_table.mat')
    data_dir =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_duels_data_wo_rescaling/'];
else
    load('benchmarks_table_rescaled.mat')
    data_dir =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_duels_data_rescaling/'];
end


objectives = benchmarks_table.fName;


nobj =numel(objectives);
seeds = 1:nreplicates;
update_period = maxiter+2;
more_repets = 0;
task = 'max';
hyps_update = 'none';
link = @normcdf;
identification = 'mu_c';
ns = 0;
for j = 1:nobj
    objective = char(objectives(j));
    
    [g, theta, model] = load_benchmarks(objective, [], benchmarks_table, rescaling, 'preference');
     close all
    for a =1:nacq
        acquisition_name = acquisition_funs{a};
        if strcmp(acquisition_name, 'PKG')
            modeltype = 'laplace';
        else
            modeltype = 'exp_prop';
        end
        model.modeltype = modeltype;
        acquisition_fun = str2func(acquisition_name);
        clear('xtrain', 'xtrain_norm', 'ctrain', 'score');
        
        filename = [data_dir,objective,'_',acquisition_name];
        
        optim = preferential_BO(g, task, identification, maxiter, nopt, ninit, update_period, hyps_update, acquisition_fun, ns);

        if more_repets
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
                [xtrain{r}, xtrain_norm{r}, ctrain{r}, score{r}, xbest{r}] = optim.optimization_loop(seed, theta, model);
            end
            %save_benchmark_results(acquisition_name, xtrain, xtrain_norm, ctrain, score, xbest, g, objective, data_dir)
        end
    end
end
 

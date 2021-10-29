%Preference learning: synthetic experiments

% function preference_learning(randinit, varargin)

add_bo_module;
close all

data_dir =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_tournaments_data/'];

acquisition_funs = {'MVT','kernelselfsparring_tour','random_acquisition_tour'};
acquisition_funs = {'batch_MUC'};
 maxiter =30; 
nreplicates = 10; 
nacq = numel(acquisition_funs);

rescaling = 0;
if rescaling ==0
    load('benchmarks_table.mat')
else
    load('benchmarks_table_rescaled.mat')
end
objectives = benchmarks_table.fName;
nobj =numel(objectives);
seeds = 1:nreplicates;
update_period = maxiter+2;
tsize= 4; %size of the tournaments
feedback = 'all'; %'all' best
more_repets= 0;
for j = 1:nobj
    objective = char(objectives(j));

     [g, theta, model] = load_benchmarks(objective, [], benchmarks_table, rescaling);
    model.nsamples= tsize;
        model.link = @normcdf;
    model.modeltype = 'exp_prop';
    model.regularization = 'nugget';

    close all
    for a =1:nacq
        acquisition_name = acquisition_funs{a};
        acquisition_fun = str2func(acquisition_name);
        clear('xtrain', 'xtrain_norm', 'ctrain', 'score');

        filename = [data_dir,objective,'_',acquisition_name, '_', feedback];

       optim = preferential_BO(g, task, identification, maxiter, nopt, ninit, update_period, hyps_update, acquisition_fun, ns, 3);

        if more_repets
            load(filename, 'experiment')
            n = numel(experiment.(['xtrain_',acquisition_name]));
            for k = 1:nreplicates
                disp(['Repetition : ', num2str(n+k)])
                seed =n+k;
                [experiment.(['xtrain_',acquisition_name]){n+k}, experiment.(['xtrain_norm_',acquisition_name]){n+k}, experiment.(['ctrain_',acquisition_name]){n+k}, experiment.(['score_',acquisition_name]){n+k}]= optim.optimization_loop(seed, theta, model);
            end
        else
            for r=1:nreplicates
                seed  = seeds(r)
                [xtrain{r}, xtrain_norm{r}, ctrain{r}, score{r}] =  optim.optimization_loop(seed, theta, model);
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
        end

        close all
        save(filename, 'experiment')
    end
end

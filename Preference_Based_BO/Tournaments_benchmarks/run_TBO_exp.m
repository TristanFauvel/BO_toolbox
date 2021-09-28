%Preference learning: synthetic experiments

% function preference_learning(randinit, varargin)

add_bo_module;
close all

data_dir =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_tournaments_data/'];

acquisition_funs = {'MVT','kernelselfsparring_tour','random_acquisition_tour'};
acquisition_funs = {'batch_MUC'};
%there is a problem with 'value_expected_improvement'
maxiter =30;%100; %total number of iterations : 200

%answer = 'max_mu_g'; %the way I report the maximum, either by maximizing the predictive mean of g, either by maximizing the soft-copeland score, EITHER BY maximizing MU_C: which makes more sense (but this depends on the acquisition function).

nreplicates = 10; %20;

nacq = numel(acquisition_funs);


% wbar = waitbar(0,'Computing...');
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
for j = 11:nobj
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

        if more_repets
            load(filename, 'experiment')
            n = numel(experiment.(['xtrain_',acquisition_name]));
            for k = 1:nreplicates
                disp(['Repetition : ', num2str(n+k)])
                seed =n+k;
                [experiment.(['xtrain_',acquisition_name]){n+k}, experiment.(['xtrain_norm_',acquisition_name]){n+k}, experiment.(['ctrain_',acquisition_name]){n+k}, experiment.(['score_',acquisition_name]){n+k}]=  TBO_loop(acquisition_fun, seed, maxiter, theta, g, update_period, model, tsize,feedback);
            end
        else
            for r=1:nreplicates
                seed  = seeds(r)
                [xtrain{r}, xtrain_norm{r}, ctrain{r}, score{r}] =  TBO_loop(acquisition_fun, seed, maxiter, theta, g, update_period, model, tsize,feedback);
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

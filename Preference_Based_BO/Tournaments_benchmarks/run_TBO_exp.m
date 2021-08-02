%Preference learning: synthetic experiments

% function preference_learning(randinit, varargin)

add_bo_module;
close all

data_dir =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_tournaments_data/'];

acquisition_funs = {'random_acquisition_pref','kernelselfsparring','MVT'};

%there is a problem with 'value_expected_improvement'
maxiter = 50;%100; %total number of iterations : 200

%answer = 'max_mu_g'; %the way I report the maximum, either by maximizing the predictive mean of g, either by maximizing the soft-copeland score, EITHER BY maximizing MU_C: which makes more sense (but this depends on the acquisition function).

nreplicates = 20; %20;

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
tsize= 3; %size of the tournaments
for j = 1:nobj 
    objective = char(objectives(j));
    
    link = @normcdf;
    modeltype = 'exp_prop';
    [g, theta, lb, ub, lb_norm, ub_norm, theta_lb, theta_ub, kernelfun, kernelname] = load_benchmarks(objective, [], benchmarks_table, rescaling);
    close all
    for a =1:nacq
        acquisition_name = acquisition_funs{a};
        acquisition_fun = str2func(acquisition_name);
        clear('xtrain', 'xtrain_norm', 'ctrain', 'score');
        
        filename = [data_dir,objective,'_',acquisition_name];
            for r=1:nreplicates 
                seed  = seeds(r)
                [xtrain{r}, xtrain_norm{r}, ctrain{r}, score{r}] =  TBO_loop(acquisition_fun, seed, lb, ub, maxiter, theta, g, update_period, modeltype, theta_lb, theta_ub, kernelname, kernelfun, lb_norm, ub_norm, link, tsize);
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

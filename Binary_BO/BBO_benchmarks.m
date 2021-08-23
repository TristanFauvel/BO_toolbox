clear all

add_bo_module;
close all

data_dir =  [pathname,'/Binary_BO/Data/'];


acquisition_funs = {'EI_Tesch', 'TS_binary','random_acquisition_binary', 'UCB_binary', 'UCB_binary_latent', 'bivariate_EI_binary'};
%Puis faire BKG 
maxiter = 50; %total number of iterations : 200

nreplicates = 20;

nacq = numel(acquisition_funs);
nbo = 2;

load('benchmarks_table.mat')
objectives = benchmarks_table.fName; %; 'Ursem_waves';'forretal08'; 'camel6';'goldpr'; 'grlee12';'forretal08'};
% objectives = {'spheref'}; %%%%%%%%%%%%%%%%%
nobj =numel(objectives);
seeds = 1:nreplicates;
update_period = maxiter+2; % do not update the hyperparameters;

rescaling = 1;
if rescaling == 0
    load('benchmarks_table.mat')
else
    load('benchmarks_table_rescaled.mat')
end


link = @normcdf;
for j = 1:nobj
    objective = char(objectives(j));    
    [g, theta, model] = load_benchmarks(objective, [], benchmarks_table, rescaling);
    close all
    
    model.link = link;
    model.type = 'classification';
    
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
            [xtrain{r}, xtrain_norm{r}, ctrain{r}, score{r}] =  BBO_loop(acquisition_fun, nbo, seed, maxiter, theta, g, update_period, model);
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

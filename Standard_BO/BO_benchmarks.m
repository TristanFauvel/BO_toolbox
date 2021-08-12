clear all
close all

add_bo_module;

figures_folder = pathname;
graphics_style_paper;

seed=1;
maxiter= 50;% 100;
rng(seed)


options_theta.method = 'lbfgs';
options_theta.verbose = 1;

update_period = 15000;

ninit = 5000;
nopt= 5;% number of time steps before starting using the acquisition functions.
nrepets = 20; %20;
seeds=1:nrepets;


load('benchmarks_table.mat')

objectives = benchmarks_table.fName;
acquisition_funs = {'random_acquisition','Thompson_sampling','EI','GP_UCB'}; %, 'EI_bin_mu', 'KG'


nobj = numel(objectives);
nacq = numel(acquisition_funs);
for j = 1:nobj
    bias = 0;
    objective = char(objectives(j));
    
    [g, theta.cov, model] = load_benchmarks(objective, [], benchmarks_table, 0);
    
    
    theta.mean = 0;
    max_g = 0;
    meanfun = @constant_mean;
    for ji = 1:nacq
        acquisition_name = acquisition_funs{ji};
        acquisition_fun = str2func(acquisition_name);
        clear('xtrain', 'xtrain_norm', 'ctrain', 'score');
        for k = 1:nrepets
            disp(['Repetition ', num2str(k)])
            seed = seeds(k);
            [xtrain{k}, xtrain_norm{k}, ytrain{k}, score{k}]= BO_loop(g, maxiter, nopt, model, theta, acquisition_fun, ninit, max_g, seed);
        end
        clear('experiment')
        fi = ['xtrain_',acquisition_name];
        experiment.(fi) = xtrain;
        fi = ['xtrain_norm_',acquisition_name];
        experiment.(fi) = xtrain_norm;
        fi = ['ctrain_',acquisition_name];
        experiment.(fi) = ytrain;
        fi = ['score_',acquisition_name];
        experiment.(fi) = score;
        
        filename = [pathname,'/Standard_BO/Data_BO/',objective,'_',acquisition_name];
        save(filename, 'experiment')
    end
end
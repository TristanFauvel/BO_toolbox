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
nrepets = 1; %20;
seeds=1:nrepets;


load('benchmarks_table.mat')

objectives = benchmarks_table.fName;
acquisition_funs = {'GP_UCB','random_acquisition','Thompson_sampling', 'EI'}; %, 'EI_bin_mu', 'KG'


nobj = numel(objectives);
nacq = numel(acquisition_funs);

task = 'max';
identification = 'mu_g';
hyps_update = 'none';
ns = 0;
noise= 0 ;

for j = 1:nobj
    bias = 0;
    objective = char(objectives(j));

    [g, theta, model] = load_benchmarks(objective, [], benchmarks_table, 0, 'regression');


    theta.mean = 0;
    max_g = 0;
    meanfun = @constant_mean;
    for ji = 1:nacq
        acquisition_name = acquisition_funs{ji};
        acquisition_fun = str2func(acquisition_name);

        optim = standard_BO(g, task, identification, maxiter, nopt, ninit, update_period, hyps_update, acquisition_fun, ns, noise);

        clear('xtrain', 'xtrain_norm', 'ctrain', 'score');
        for k = 1:nrepets
            disp(['Repetition ', num2str(k)])
            seed = seeds(k);
            [xtrain{k}, xtrain_norm{k}, ytrain{k}, score{k}, xbest{k}]= optim.optimization_loop(seed, theta, model);
        end
       save_benchmark_results(acquisition_name, structure_name, xtrain, xtrain_norm, ytrain, score, xbest, g, objective, data_dir)
    end
end

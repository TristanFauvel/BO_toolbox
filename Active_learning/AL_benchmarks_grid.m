clear all
close all

add_bo_module;

seed=1;
maxiter= 50;% 100;
rng(seed)


options_theta.method = 'lbfgs';
options_theta.verbose = 1;

update_period = maxiter+ 2;

ninit = maxiter +2 ;
nopt= 5;% number of time steps before starting using the acquisition functions.
nrepets = 20; %20;
seeds=1:nrepets;


load('active_learning_benchmarks_table.mat')

objectives = benchmarks_table.fName;
acquisition_funs = {'BALD', 'TME', 'maxvar', 'random'}; 


nobj = numel(objectives);
nacq = numel(acquisition_funs);
for j = 1:nobj
    bias = 0;
    objective = char(objectives(j));
    
    [y, theta.cov, lb, ub, theta_lb, theta_ub,kernelfun, kernelname] = load_benchmarks(objective, [], benchmarks_table);
    theta.mean = 0;
    meanfun = @constant_mean;
    for ji = 1:nacq
        acquisition_name = acquisition_funs{ji};
        acquisition_fun = str2func(acquisition_name);
        clear('xtrain', 'ytrain', 'cum_regret', 'score');
        for k = 1:nrepets
            disp(['Repetition ', num2str(k)])
            seed = seeds(k);
            [xtrain{k}, ytrain{k}, cum_regret{k}, score{k}]= AL_loop_grid(x, y, maxiter, nopt, kernelfun, meanfun, theta, acquisition_fun, ninit, theta_lb, theta_ub, lb, ub, seed);
        end
        clear('experiment')
        fi = ['xtrain_',acquisition_name];
        experiment.(fi) = xtrain;
        fi = ['xtrain_norm_',acquisition_name];
        experiment.(fi) = xtrain_norm;
        fi = ['ytrain_',acquisition_name];
        experiment.(fi) = ytrain;
        fi = ['score_',acquisition_name];
        experiment.(fi) = score;
        
        filename = [pathname,'/Active_learning/Data_active_learning_grid/',objective,'_',acquisition_name];
        save(filename, 'experiment')
    end
end
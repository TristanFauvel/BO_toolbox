add_bo_module
data_dir =  [pathname,'/Standard_BO/Data_BO'];
figure_folder = [pathname,'/Standard_BO/Figures/'];
figname =  'Standard_BO_scores_benchmarks';

acquisition_funs = {'EI','GP_UCB','Thompson_sampling', 'random_acquisition'}; %, 'EI_bin_mu', 'KG'
names = {'EI','GP-UCB','Thompson sampling', 'Random'};

load('benchmarks_table.mat')
objectives = benchmarks_table.fName; %; 'Ursem_waves';'forretal08'; 'camel6';'goldpr'; 'grlee12';'forretal08'};
objectives_names = benchmarks_table(benchmarks_table.fName == objectives,:).Name; %{'GP1d','Forrester (2008)', 'Gramacy and Lee (2012)', 'Levy', 'Goldstein-Price', 'Six Hump Camel', 'Ursem-Waves'};
nobj =numel(objectives);


nreps = 20;
maxiter = 50;
t = ranking_analysis(data_dir, names, objectives, acquisition_funs, nreps, maxiter);

table2latex(t, 'BO_benchmark_results')

objectives = categorical({'Ursem_waves';'forretal08'; 'camel6';'goldpr'; 'grlee12'}); 
objectives_names = benchmarks_table(any(benchmarks_table.fName == objectives',2),:).Name; 
plot_optimalgos_comparison(objectives, objectives_names, acquisition_funs, names, figure_folder,data_dir, figname, nreps, maxiter)




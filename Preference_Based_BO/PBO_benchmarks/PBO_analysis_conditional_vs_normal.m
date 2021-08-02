add_bo_module;
data_dir_w =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_duels_data/'];
data_dir_wo =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_PBO_wo_condition'];

figure_folder = [pathname,'/Preference_Based_BO/Figures/'];
figname =  'PBO_scores_benchmarks_w_wo_conditioning';

acquisition_funs = {'maxvar_challenge'};

names = {'Maximum Variance Challenge'};

load('benchmarks_table.mat')
objectives = benchmarks_table.fName; %; 'Ursem_waves';'forretal08'; 'camel6';'goldpr'; 'grlee12';'forretal08'};
objectives_names = benchmarks_table.fName; %{'GP1d','Forrester (2008)', 'Gramacy and Lee (2012)', 'Levy', 'Goldstein-Price', 'Six Hump Camel', 'Ursem-Waves'};
nobj =numel(objectives);

nreps = 20;
maxiter = 50;
t = ranking_analysis_w_wo_condition(data_dir_w, data_dir_wo, names, objectives, acquisition_funs, nreps, maxiter);

table2latex(t, [figure_folder, '/PBO_benchmark_results_w_wo_conditioning'])


objectives = categorical({'Ursem_waves';'forretal08'; 'camel6';'goldpr'; 'grlee12'}); 
objectives_names = benchmarks_table(any(benchmarks_table.fName == objectives',2),:).Name; 
plot_optimalgos_comparison(objectives, objectives_names, acquisition_funs, names, figure_folder,data_dir, figname)

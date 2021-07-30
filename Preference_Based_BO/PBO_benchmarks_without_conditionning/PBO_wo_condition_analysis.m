data_dir_w =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_PBO'];
data_dir_wo =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_PBO_wo_condition'];

% figure_folder = [pathname,'/Preference_Based_BO/Figures/'];
% figname =  'PBO_scores_benchmarks';

acquisition_funs = {'maxvar_challenge'};
names = {'Maximum Variance Challenge'};

load('benchmarks_table.mat')
objectives = benchmarks_table.fName; %; 'Ursem_waves';'forretal08'; 'camel6';'goldpr'; 'grlee12';'forretal08'};
objectives_names = benchmarks_table.fName; %{'GP1d','Forrester (2008)', 'Gramacy and Lee (2012)', 'Levy', 'Goldstein-Price', 'Six Hump Camel', 'Ursem-Waves'};
nobj =numel(objectives);


plot_optimalgos_comparison(objectives, objectives_names, acquisition_funs, names, figure_folder,data_dir, figname)
nreps = 20;
maxiter = 50;
t = ranking_analysis_w_vs_wo_condition(data_dir_w, data_dir_wo, names, objectives, acquisition_funs, nreps, maxiter);

table2latex(t, 'PBO_benchmark_results')





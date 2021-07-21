data_dir =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_duels_data'];
figure_folder = [pathname,'/Preference_Based_BO/Figures/'];
figname =  'PBO_scores_benchmarks';

acquisition_funs = {'DTS','random_acquisition_pref','kernelselfsparring','maxvar_challenge', 'bivariate_EI', 'Brochu_EI', 'Thompson_challenge'};
names = {'DTS','Random', 'KSS', 'MVC', 'Bivariate EI (Nielsen 2015)', 'EI (Brochu 2010)', 'Thompson Challenge',};
names = {'Duel Thompson Sampling','Random', 'Kernel Self-Sparring', 'Maximum Variance Challenge', 'Bivariate Expected Improvement (Nielsen 2015)', 'Expected Improvement (Brochu 2010)', 'Thompson Challenge'};

load('benchmarks_table.mat')
objectives = benchmarks_table.fName; %; 'Ursem_waves';'forretal08'; 'camel6';'goldpr'; 'grlee12';'forretal08'};
objectives_names = benchmarks_table.fName; %{'GP1d','Forrester (2008)', 'Gramacy and Lee (2012)', 'Levy', 'Goldstein-Price', 'Six Hump Camel', 'Ursem-Waves'};
nobj =numel(objectives);


plot_optimalgos_comparison(objectives, objectives_names, acquisition_funs, names, figure_folder,data_dir, figname)
nreps = 1;
maxiter = 10;
t = ranking_analysis(data_dir, names, objectives, acquisition_funs, nreps, maxiter);

table2latex(t, 'PBO_benchmark_results')





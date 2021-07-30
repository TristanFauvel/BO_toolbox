add_bo_module;
data_dir =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_duels_data'];
figure_folder = [pathname,'/Preference_Based_BO/Figures/'];
figname =  'PBO_scores_benchmarks';

acquisition_funs = {'EIIG', 'Dueling_UCB', 'MaxEntChallenge','DTS','random_acquisition_pref','kernelselfsparring','maxvar_challenge', 'bivariate_EI', 'Brochu_EI', 'Thompson_challenge'};

names = {'EIIG (Benavoli 2020)', 'Dueling UCB (Benavoli 2020)', 'MaxEntChallenge','Duel Thompson Sampling (modified from Gonzalez 2017)','Random', 'Kernel Self-Sparring (Sui 2017)', 'Maximum Variance Challenge', 'Bivariate Expected Improvement (Nielsen 2015)', 'Expected Improvement (Brochu 2010)', 'Dueling Thompson (Benavoli 2020)'};

load('benchmarks_table.mat')
objectives = benchmarks_table.fName; %; 'Ursem_waves';'forretal08'; 'camel6';'goldpr'; 'grlee12';'forretal08'};
objectives_names = benchmarks_table.fName; %{'GP1d','Forrester (2008)', 'Gramacy and Lee (2012)', 'Levy', 'Goldstein-Price', 'Six Hump Camel', 'Ursem-Waves'};
nobj =numel(objectives);

nreps = 20;
maxiter = 50;
t = ranking_analysis(data_dir, names, objectives, acquisition_funs, nreps, maxiter);

table2latex(t, [figure_folder, '/PBO_benchmark_results'])


objectives = categorical({'Ursem_waves';'forretal08'; 'camel6';'goldpr'; 'grlee12'}); 
objectives_names = benchmarks_table(any(benchmarks_table.fName == objectives',2),:).Name; 
plot_optimalgos_comparison(objectives, objectives_names, acquisition_funs, names, figure_folder,data_dir, figname)





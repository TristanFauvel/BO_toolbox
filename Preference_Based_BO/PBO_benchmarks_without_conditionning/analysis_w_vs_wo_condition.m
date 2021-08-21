add_bo_module;

pathname = '/home/tfauvel/Documents/BO_toolbox';
data_dir_w =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_duels_data'];
data_dir_wo =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_PBO_wo_condition'];
figure_folder = [pathname,'/Preference_Based_BO/Figures/'];
folder = '/home/tfauvel/Documents/BO_toolbox/Preference_Based_BO/PBO_benchmarks/Figures';

% figure_folder = [pathname,'/Preference_Based_BO/Figures/'];
% figname =  'PBO_scores_benchmarks';

acq_funs = {'maxvar_challenge'};
 acq_funs = {'Dueling_UCB'};
 
 load('/home/tfauvel/Documents/BO_toolbox/Acquisition_funs_table','T')
acquisition_funs = cellstr(char(T(any(T.acq_funs == acq_funs,2),:).acq_funs)); 
acquisition_names = char(T(any(T.acq_funs == acq_funs,2),:).names); 
acquisition_names_citation = char(T(any(T.acq_funs == acq_funs,2),:).names_citations); 
short_acq_names= char(T(any(T.acq_funs == acq_funs,2),:).short_names); 

% acquisition_funs = {'kernelselfsparring'};
% names = {'KSS'};

load('benchmarks_table.mat')
objectives = benchmarks_table.fName; %; 'Ursem_waves';'forretal08'; 'camel6';'goldpr'; 'grlee12';'forretal08'};
objectives_names = benchmarks_table.Name; %{'GP1d','Forrester (2008)', 'Gramacy and Lee (2012)', 'Levy', 'Goldstein-Price', 'Six Hump Camel', 'Ursem-Waves'};
objectives = categorical(cellstr('grlee12')); %; 'Ursem_waves';'forretal08'; 'camel6';'goldpr'; 'grlee12';'forretal08'};
objectives_names = categorical(cellstr('Gramacy and Lee (2012)')); %{'GP1d','Forrester (2008)', 'Gramacy and Lee (2012)', 'Levy', 'Goldstein-Price', 'Six Hump Camel', 'Ursem-Waves'};

nobj =numel(objectives);

nreps = 20;
maxiter = 50;
t = ranking_analysis_w_vs_wo_condition(data_dir_w, data_dir_wo, objectives_names, objectives, acquisition_funs, nreps, maxiter);
table2latex(t, 'PBO_benchmark_results_w_vs_wo')

objectives = {'forretal08', 'grlee12', 'levy', 'goldpr', 'camel6', 'Ursem_waves'};
objectives_names = {'Forrester (2008)', 'Gramacy and Lee (2012)', 'Levy', 'Goldstein-Price', 'Six Hump Camel', 'Ursem-Waves'};
rescaling = 0;

figname  = 'PBO_w_vs_wo';
 
plot_optimalgos_comparison_w_vs_wo(objectives, objectives_names, acquisition_funs, names, figure_folder,data_dir_w, data_dir_wo, figname, nreps, maxiter, rescaling)



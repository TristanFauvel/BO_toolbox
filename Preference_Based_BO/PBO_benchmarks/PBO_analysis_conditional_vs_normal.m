add_bo_module;
data_dir_w =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_duels_data/'];
data_dir_wo =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_PBO_wo_condition'];

figure_folder = [pathname,'/Preference_Based_BO/Figures/'];
figname =  'PBO_scores_benchmarks_w_wo_conditioning';

acq_funs = {'maxvar_challenge'};

load('/home/tfauvel/Documents/BO_toolbox/Acquisition_funs_table','T')
acquisition_funs = cellstr(char(T(any(T.acq_funs == acq_funs,2),:).acq_funs)); 
acquisition_names = char(T(any(T.acq_funs == acq_funs,2),:).names); 
acquisition_names_citation = char(T(any(T.acq_funs == acq_funs,2),:).names_citations); 
short_acq_names= char(T(any(T.acq_funs == acq_funs,2),:).short_names); 


 
load('benchmarks_table.mat')
objectives = benchmarks_table.fName; %; 'Ursem_waves';'forretal08'; 'camel6';'goldpr'; 'grlee12';'forretal08'};
objectives_names = benchmarks_table.fName; %{'GP1d','Forrester (2008)', 'Gramacy and Lee (2012)', 'Levy', 'Goldstein-Price', 'Six Hump Camel', 'Ursem-Waves'};
nobj =numel(objectives);

nreps = 20;
maxiter = 50;
t = ranking_analysis_w_wo_condition(data_dir_w, data_dir_wo, acquisition_names_citation, objectives, acquisition_funs, nreps, maxiter);

table2latex(t, [figure_folder, '/PBO_benchmark_results_w_wo_conditioning'])


objectives = categorical({'Ursem_waves';'forretal08'; 'camel6';'goldpr'; 'grlee12'}); 
objectives_names = benchmarks_table(any(benchmarks_table.fName == objectives',2),:).Name; 

rescaling = 1;
plot_optimalgos_comparison_w_vs_wo(objectives, objectives_names, acquisition_funs, ...
    acquisition_names, figure_folder,data_dir_w, data_dir_wo, figname, nreps, maxiter, rescaling)
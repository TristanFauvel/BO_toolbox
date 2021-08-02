data_dir =  ['/home/tfauvel/Documents/BO_toolbox/Active_learning/', 'Data_active_learning_preference'];

% figure_folder = [pathname,'/Preference_Based_BO/Figures/'];
% figname =  'PBO_scores_benchmarks';

acquisition_funs = {'BALD'};
 acquisition_name = 'BALD';
% acquisition_funs = {'kernelselfsparring'};
% names = {'KSS'};

load('benchmarks_table.mat')
objectives = benchmarks_table.fName;  
objectives_names = benchmarks_table(benchmarks_table.fName == objectives,:).Name;  
nobj =numel(objectives);

nreps = 20;
maxiter = 50;
t =  ranking_analysis_AL_w_vs_wo_condition(data_dir, [], objectives, acquisition_funs, nreps, maxiter);
table2latex(t, 'PBO_benchmark_results')

objectives = {'forretal08', 'grlee12', 'levy', 'goldpr', 'camel6', 'Ursem_waves'};
objectives_names = {'Forrester (2008)', 'Gramacy and Lee (2012)', 'Levy', 'Goldstein-Price', 'Six Hump Camel', 'Ursem-Waves'};

objectives = {'ackley'};
objectives_names = {'Ackley'};

rescaling = 0;
figure_folder = [];
figname = [];
plot_AL_comparison_w_vs_wo(objectives, objectives_names, acquisition_funs, [], figure_folder,data_dir, figname, nreps, maxiter, rescaling)
% 
% 

data_dir =  [pathname,'/Binary_BO/Data/'];
figure_folder = [pathname,'/Binary_BO/Figures/'];
figname =  'PBO_scores_benchmarks';

objectives = {'GP1d', 'forretal08', 'grlee12', 'levy', 'goldpr', 'camel6', 'Ursem_waves'};
objectives_names = {'GP1d','Forrester (2008)', 'Gramacy and Lee (2012)', 'Levy', 'Goldstein-Price', 'Six Hump Camel', 'Ursem-Waves'};
 acq_funs = {'TS_binary', 'KG_binary', 'random_acquisition_binary'};
load('/home/tfauvel/Documents/BO_toolbox/Acquisition_funs_table','T')
acquisition_funs = cellstr(char(T(any(T.acq_funs == acq_funs,2),:).acq_funs)); 
acquisition_names = char(T(any(T.acq_funs == acq_funs,2),:).names); 
acquisition_names_citation = char(T(any(T.acq_funs == acq_funs,2),:).names_citations); 
short_acq_names= char(T(any(T.acq_funs == acq_funs,2),:).short_names); 


nreps = 40;
maxiter= 60;
plot_optimalgos_comparison(objectives, objectives_names, acquisition_funs, names, figure_folder,data_dir, figname, nreps, maxiter)

t1 = ranking_analysis(data_dir, names, objectives(1:3), acquisition_funs, nreps, maxiter);

t2 = ranking_analysis(data_dir, names(1:2), objectives, acquisition_funs([1,3]), nreps, maxiter);


table2latex(t, 'BBO_benchmark_results')






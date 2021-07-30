add_bo_module
data_dir =  [pathname,'/Active_learning/Data_active_learning_binary_grid'];
figure_folder = [pathname,'/Active_learning/Figures_active_learning/'];
figname =  'Active_learning_binary_grid_benchmarks';
acquisition_funs = {'TME_sampling_binary', 'BALD_grid'};
names = {'TME_sampling_binary','BALD_grid'};

nobj = 4;
fun = cell(1,nobj);
for i = 1:nobj
    fun{i} = 'japan';
end

kernelnames = {'Matern32', 'Matern32','Matern52','Matern52'};
lengthscales = {'long', 'short','long', 'short'};

objectives = cell(1,nobj);

for j =1:nobj
    objectives{j} = [fun{j}, '_', kernelnames{j}, '_',lengthscales{j}];
end


nreps = 10;
maxiter = 35;
t = ranking_analysis_AL_grid(data_dir, names, objectives, acquisition_funs, nreps, maxiter);

table2latex(t, 'Active_learning_grid_benchmark_results')

scaling = 1;
objectives_names = objectives; 
plot_optimalgos_comparison_AL_grid(objectives, objectives_names, acquisition_funs, names, figure_folder,data_dir, figname, nreps, maxiter, scaling)
 







       

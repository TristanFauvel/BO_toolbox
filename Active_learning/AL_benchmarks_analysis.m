add_bo_module
data_dir =  [pathname,'/Active_learning/Data_active_learning_binary_grid'];
figure_folder = [pathname,'/Active_learning/Figures_active_learning/'];
figname =  'Active_learning_binary_grid_benchmarks';
acquisition_funs = {'TME_sampling_binary', 'maxvar_binary_grid', 'random','BALD_grid'};
names = {'TME_sampling_binary', 'maxvar_binary_grid', 'random','BALD_grid'};

nobj = 6;
fun = cell(1,nobj);
for i = 1:nobj
    fun{i} = 'japan';
end
kernelnames = {'Matern32', 'Matern32','Matern52','Matern52','Gaussian','Gaussian'};
lengthscales = {'long', 'short','long', 'short','long', 'short'};

objectives = cell(1,nobj);

for j =1:nobj
objectives{j} = [fun{j}, '_', kernelnames{j}, '_',lengthscales{j}];
end


nreps = 5;
maxiter = 30;
t = ranking_analysis(data_dir, names, objectives, acquisition_funs, nreps, maxiter);

table2latex(t, 'Active_learning_grid_benchmark_results')

objectives_names = objectives; 
plot_optimalgos_comparison(objectives, objectives_names, acquisition_funs, names, figure_folder,data_dir, figname, nreps, maxiter)
 







       

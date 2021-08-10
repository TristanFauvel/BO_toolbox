add_bo_module;
data_dir =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_duels_data'];
figure_folder = [pathname,'/Preference_Based_BO/Figures/'];
figname =  'PBO_scores_benchmarks';

acq_funs = {'EIIG', 'Dueling_UCB', 'MaxEntChallenge','DTS','random_acquisition_pref','kernelselfsparring','maxvar_challenge', 'bivariate_EI', 'Brochu_EI', 'Thompson_challenge'};

load('/home/tfauvel/Documents/BO_toolbox/Acquisition_funs_table','T')
acquisition_funs = cellstr(char(T(any(T.acq_funs == acq_funs,2),:).acq_funs)); 
acquisition_names = char(T(any(T.acq_funs == acquisition_funs,2),:).names); 
acquisition_names_citation = char(T(any(T.acq_funs == acquisition_funs,2),:).names_citations); 
short_acq_names= T(any(T.acq_funs == acquisition_funs,2),:).short_names; 


load('benchmarks_table.mat')
objectives = benchmarks_table.fName; 
objectives_names = benchmarks_table.fName; 
nobj =numel(objectives);

nreps = 20;
maxiter = 50;
[t, Best_ranking, AUC_ranking,b] = ranking_analysis(data_dir, acquisition_names_citation, objectives, acquisition_funs, nreps, maxiter, []);

table2latex(t, [figure_folder, '/PBO_benchmark_results'])

rescaling = 0;
objectives = categorical({'Ursem_waves';'forretal08'; 'camel6';'goldpr'; 'grlee12'}); 


objectives_names = benchmarks_table(any(benchmarks_table.fName == objectives',2),:).Name; 
fig = plot_optimalgos_comparison(objectives, objectives_names, acquisition_funs, acquisition_names, figure_folder,data_dir, figname, nreps, maxiter, rescaling);

figname  = 'optim';
folder = ['/home/tfauvel/Documents/BO_toolbox/Preference_Based_BO/PBO_benchmarks/',figname];
savefig(fig, [folder, figname, '.fig'])
exportgraphics(fig, [folder,figname, '.pdf']);
exportgraphics(fig, [folder, figname, '.png'], 'Resolution', 300);



mr = 2;
mc= 1;
legend_pos = [-0.1,1];
i=0;
graphics_style_paper;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth(mc) fheight(mr)]);
fig.Color =  [1 1 1];
tiledlayout(mr, mc, 'TileSpacing', 'compact', 'padding','compact');
nexttile()
mat = flipud(Best_ranking);
p =  plot_matrix(mat, {}, short_acq_names(b,:));
i=i+1;
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)


nexttile()
mat = flipud(AUC_ranking);
p =  plot_matrix(mat, short_acq_names(b,:), short_acq_names(b,:));
i=i+1;
text(legend_pos(1), legend_pos(2),['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)

% colormap(cmap)

figname  = 'Matrices';
savefig(fig, [figure_folder, figname, '.fig'])
exportgraphics(fig, [figure_folder, figname, '.pdf']);
exportgraphics(fig, [figure_folder, figname, '.png'], 'Resolution', 300);



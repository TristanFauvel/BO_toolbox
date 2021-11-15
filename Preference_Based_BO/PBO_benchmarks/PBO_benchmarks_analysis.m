add_bo_module;
figure_folder = [pathname,'/Preference_Based_BO/Figures/'];
folder = '/home/tfauvel/Documents/BO_toolbox/Preference_Based_BO/PBO_benchmarks/Figures';

rescaling = 0;
if rescaling ==0
    data_dir =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_duels_data_wo_rescaling/'];
else
    data_dir =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_duels_data_rescaling/'];
end
% data_dir =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_duels_data'];

figname =  'PBO_scores_benchmarks';

all_acq_funs = {'EIIG', 'Dueling_UCB',  'DTS','random_acquisition_pref','kernelselfsparring','MUC', 'bivariate_EI', 'Brochu_EI', 'Thompson_challenge'};

acq_funs = all_acq_funs;
load('/home/tfauvel/Documents/BO_toolbox/Acquisition_funs_table','T')
acquisition_funs = cellstr(char(T(any(T.acq_funs == acq_funs,2),:).acq_funs)); 
acquisition_names = char(T(any(T.acq_funs == acq_funs,2),:).names); 
acquisition_names_citation = char(T(any(T.acq_funs == acq_funs,2),:).names_citations); 
short_acq_names= char(T(any(T.acq_funs == acq_funs,2),:).short_names); 


load('benchmarks_table.mat')
objectives = benchmarks_table.fName; 
objectives_names = benchmarks_table.Name; 
nobj =numel(objectives);

nreps = 20;
maxiter = 50;
% nreps = 40;
% maxiter = 80;
% 
[t, Best_ranking, AUC_ranking,b,signobj,ranking, final_values, AUCs] = ranking_analysis(data_dir,...
    acquisition_names_citation, objectives, acquisition_funs , nreps, [], [], 'max', 'score');

table2latex(t,[folder,'/PBO_benchmarks_results'])

mr = 2;
mc= 1;
legend_pos = [-0.1,1];
i=0;
graphics_style_paper;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth(mc) fheight(mr)]);
fig.Color =  background_color;
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

figname  = 'PBO_Matrices';
figure_file = [folder,'/' figname];
savefig(fig, [figure_file,'.fig'])
exportgraphics(fig, [figure_file, '.pdf']);
exportgraphics(fig, [figure_file, '.png'], 'Resolution', 300);



%%
% acquisition_funs = acquisition_funs(s); 
% acquisition_names = acquisition_names(s,:); 
% acquisition_names_citation = acquisition_names_citation(s,:); 
% short_acq_names = short_acq_names(s,:);
acq_funs = {'Dueling_UCB', 'MUC', 'bivariate_EI', 'DTS', 'Thompson_challenge'};
acquisition_funs = cellstr(char(T(any(T.acq_funs == acq_funs,2),:).acq_funs)); 
acquisition_names = char(T(any(T.acq_funs == acq_funs,2),:).names); 
acquisition_names_citation = char(T(any(T.acq_funs == acq_funs,2),:).names_citations); 
short_acq_names= char(T(any(T.acq_funs == acq_funs,2),:).short_names); 

[~,~,~,~,signobj] = ranking_analysis(data_dir, acquisition_names_citation, objectives, acquisition_funs, nreps, [], [], 'max', 'score');
% s = [14, 18, 34, 30, 28];
s =[1,14,34];

objectives = benchmarks_table.fName; 
objectives_names = benchmarks_table.Name; 
objectives = objectives(s);
objectives_names = objectives_names(s);

% objectives_names = benchmarks_table(any(benchmarks_table.fName == objectives',2),:).Name;
clear('lines')
% lines = cell(size(acquisition_funs'));
% lines(:) = {'-'};
lines(:) = {'-',':',':',':',':'};

fig = plot_optimalgos_comparison(objectives, objectives_names, acquisition_funs, char(acquisition_names), figure_folder,data_dir, figname, nreps, maxiter, rescaling, [], [], 'max', 'score',lines);

figname  = 'optim_trajectories_PBO';
figure_file = [folder,'/' figname];
savefig(fig, [figure_file,'.fig'])
exportgraphics(fig, [figure_file, '.pdf']);
exportgraphics(fig, [figure_file, '.png'], 'Resolution', 300);



legends = T(any(T.acq_funs == acq_funs,2),:).short_names; 
fig =  plot_benchmarks_histograms(final_values, AUCs, legends, b);

pathname = '/home/tfauvel/Documents/BO_toolbox';
 data_dir =  [pathname,'/Binary_BO/Data'];
 
figure_folder = [pathname,'/Binary_BO/Figures/'];
figname =  'PBO_scores_benchmarks';

load('benchmarks_table.mat')
objectives = benchmarks_table.fName; 
objectives_names = benchmarks_table.Name; 
 

nobj =numel(objectives);


 
 %'BKG'
%all_acq_funs = {'TS_binary', 'random_acquisition_binary','UCB_binary', 'UCB_binary_latent', 'EI_Tesch', 'BKG'};
all_acq_funs = {'TS_binary','random_acquisition_binary','UCB_binary', 'UCB_binary_latent', 'EI_Tesch'};

acq_funs = all_acq_funs;
load('/home/tfauvel/Documents/BO_toolbox/Acquisition_funs_table','T')
acquisition_funs = cellstr(char(T(any(T.acq_funs == acq_funs,2),:).acq_funs)); 
acquisition_names = char(T(any(T.acq_funs == acq_funs,2),:).names); 
acquisition_names_citation = char(T(any(T.acq_funs == acq_funs,2),:).names_citations); 
short_acq_names= char(T(any(T.acq_funs == acq_funs,2),:).short_names); 

 
nreps = 60;

maxiter= 100;
optim = 'max_proba';
score_measure = 'score_c';
% optim = 'max';
[t, Best_ranking, AUC_ranking,b, signobj] = ranking_analysis(data_dir, ...
    char(acquisition_names_citation), objectives, acquisition_funs, nreps,[],[], optim,score_measure);

table2latex(t, [figure_folder,'BBO_benchmark_results'])

%%
% acq_funs = all_acq_funs;
% acq_funs = acq_funs(b); %plot only the n best acquisition functions
% acquisition_funs = cellstr(char(T(any(T.acq_funs == acq_funs,2),:).acq_funs)); 
% acquisition_names = char(T(any(T.acq_funs == acq_funs,2),:).names); 
% acquisition_names_citation = char(T(any(T.acq_funs == acq_funs,2),:).names_citations); 
% short_acq_names= char(T(any(T.acq_funs == acq_funs,2),:).short_names); 
% [~,~,~,~,signobj] = ranking_analysis(data_dir, acquisition_names_citation, objectives, acquisition_funs, nreps, maxiter, []);

objectives = benchmarks_table.fName; 
objectives_names = benchmarks_table.Name; 

%selection = signobj(1:5);
%selection = [11,16,31,32, 34];
selection = [31,32, 34];
objectives = objectives(selection,:);
objectives_names = benchmarks_table(selection,:).Name; 
 
rescaling = 1;
figname  = 'optim_trajectories_BBO';
lines = cell(size(acquisition_funs'));
 %lines(:) = {'-','-',':',':', ':', ':'};
lines(:) = {'-',':',':', ':', ':'};

plot_optimalgos_comparison(objectives, objectives_names, acquisition_funs, ...
    char(acquisition_names), figure_folder,data_dir, figname, nreps, maxiter,rescaling, [], [], 'max', score_measure, lines)



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

figname  = 'BBO_Matrices';
savefig(fig, [figure_folder, figname, '.fig'])
exportgraphics(fig, [figure_folder, figname, '.pdf']);
exportgraphics(fig, [figure_folder, figname, '.png'], 'Resolution', 300);



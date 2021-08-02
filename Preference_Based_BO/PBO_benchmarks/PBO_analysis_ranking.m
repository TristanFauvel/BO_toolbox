close all
cd(' C:\Users\tfauvel\Documents\Retinal_prosthetic_optimization\Retinal_Prosthetic_Optimization_Code\Experiment_p2p_preference\Run')
add_modules
data_dir =  [code_directory,'\Preference_Based_BO_toolbox\Data'];
cd([code_directory,'\Preference_Based_BO_toolbox'])
pathname = [data_dir,'/synthetic_exp_duels_data/'];
figures_folder = [pathname, 'figures/'];


objectives = {'GP1d', 'forretal08', 'grlee12', 'levy', 'goldpr', 'camel6', 'Ursem_waves'};
objectives_names = {'GP1d','Forrester (2008)', 'Gramacy and Lee (2012)', 'Levy', 'Goldstein-Price', 'Six Hump Camel', 'Ursem-Waves'};

nobj = numel(objectives);

acquisition_funs = {'random_acquisition_pref', 'new_DTS','active_sampling', 'MES', 'brochu_EI', 'bivariate_EI', 'random', 'decorrelatedsparring', 'kernelselfsparring'};
acquisition_funs = {'DTS','random_acquisition_pref','kernelselfsparring','maxvar_challenge', 'bivariate_EI', 'Brochu_EI', 'Thompson_challenge'};

names = {'DTS','Random', 'KSS', 'MVC', 'Bivariate EI (Nielsen 2015)', 'EI (Brochu 2010)', 'Thompson Challenge',};
names = {'Duel Thompson Sampling','Random', 'Kernel Self-Sparring', 'Maximum Variance Challenge', 'Bivariate Expected Improvement (Nielsen 2015)', 'Expected Improvement (Brochu 2010)', 'Thompson Challenge'};

nacq = numel(acquisition_funs);
graphics_style_paper;

mr = 4;
mc = 2;
fig=figure('units','centimeters','outerposition',1+[0 0 width height(mr)]);
fig.Color =  [1 1 1];
tiledlayout(mr, mc, 'TileSpacing', 'tight', 'padding','compact');
options.handle = fig;
options.alpha = 0; %0.2;
options.error= 'sem';
options.line_width = linewidth/2;
options.semilogy = false;
options.cmap = C;

rng(1);
% colors = rand(nacq,3);
% options.colors = colors;
ninit = 5;
maxiter = 100;
for j = 1:nobj
    nexttile
    objective = objectives{j};
    
    if strcmp(objective, 'goldpr')
        options.semilogy = true; %true;
    else
        options.semilogy = false;
    end
    for a = 1:nacq
        acquisition = acquisition_funs{a};
        filename = [pathname,'\',objective, '_',acquisition];
        load(filename, 'experiment');
        UNPACK_STRUCT(experiment, false)
        legends{a}=[names{a}];
        n=['a',num2str(a)];
        
        scores{a} = cell2mat(eval(['score_', acquisition])');
        
    end
    
    [ranks, average_ranks]= compute_rank(scores, ninit);
    %
    %     plots =  plot_areaerrorbar_grouped(ranks, options);
    %     box off
    %     ylabel('Rank');
    %     title(objectives_names{j})
    %      set(gca, 'Fontsize', Fontsize, 'Xlim', [ninit+1, maxiter]);
    grouporder=names;
    
    data = cell2mat(average_ranks);
    n = size(average_ranks{1},2);
%     data = rank_matrix;
%     n = size(rank_matrix,2);

    type= names;
    type= repmat(type, n,1);
    type = type(:);
    violinplot(data',type,'GroupOrder',grouporder, 'ShowData', false);
    
end
legend(plots, legends, 'Fontsize', Fontsize);
legend boxoff
xlabel('Iteration \#')

axes1 = gca;
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.53963583333902 0.062422587936693 0.444441641328221 0.17799635257443]);

set(gca, 'Fontsize', Fontsize)


figname =  'PBO_ranking_benchmarks';
folder = ['C:\Users\tfauvel\Documents\PhD\Figures\Thesis_figures\Chapter_1\',figname];
savefig(fig, [folder,'\', figname, '.fig'])
exportgraphics(fig, [folder,'\' , figname, '.pdf']);
exportgraphics(fig, [folder,'\' , figname, '.png'], 'Resolution', 300);

figure()
histogram(scores{a})





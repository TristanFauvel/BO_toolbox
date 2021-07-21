clear all
close all
cd(' C:\Users\tfauvel\Documents\Retinal_prosthetic_optimization\Retinal_Prosthetic_Optimization_Code\Experiment_p2p_preference\Run')
add_modules
data_dir =  [code_directory,'\Preference_Based_BO_toolbox\Data'];
cd([code_directory,'\Preference_Based_BO_toolbox'])
pathname = [data_dir,'/synthetic_exp_duels_data/'];
pathname_wo_condition = [data_dir,'/synthetic_exp_duels_data_without_conditioning/'];

figures_folder = [pathname, 'figures/'];


objectives = {'GP1d', 'forretal08', 'grlee12', 'levy', 'goldpr', 'camel6'};
objectives = {'forretal08', 'grlee12', 'GP1d', 'levy', 'goldpr', 'camel6'};
nobj = numel(objectives);

acquisition_funs = {'random_acquisition_pref', 'new_DTS','active_sampling', 'MES', 'random', 'decorrelatedsparring', 'kernelselfsparring'};
acquisition_funs = {'DTS','random_acquisition_pref','kernelselfsparring','maxvar_challenge', 'Thompson_challenge'};
% Warning: to compare with Brochu_EI and Bivariate_EI, I would need to
% rerun these benchmarks without conditioning (seed problem)
names = {'DTS','Random', 'KSS', 'MVC', 'Thompson Challenge',};
names = {'Duel Thompson Sampling','Random', 'Kernel Self-Sparring', 'Maximum Variance Challenge','Thompson Challenge'};

nacq = numel(acquisition_funs);
graphics_style_paper;

mc = 2;
mr = 4;
fig=figure('units','centimeters','outerposition',1+[0 0 width height(mr)]);
fig.Color =  [1 1 1];
tiledlayout(mr, mc, 'TileSpacing', 'tight', 'padding','compact');
options.handle = fig;
options.alpha = 0.2;
options.error= 'sem';
options.line_width = linewidth/2;
options.semilogy = false;
options.cmap = C;

rng(1);
% colors = rand(nacq,3);
% options.colors = colors;
for j = 1:nobj
    nexttile
    objective = objectives{j};
    
    
        options.semilogy = false;
    for a = 1:nacq
        legends{a}=[names{a}];
        n=['a',num2str(a)];
        acquisition = acquisition_funs{a};

        filename = [pathname,'\',objective, '_',acquisition];
        load(filename, 'experiment');
        UNPACK_STRUCT(experiment, false)
        scores{a} = cell2mat(eval(['score_', acquisition])');

        filename = [pathname_wo_condition,'\',objective, '_',acquisition];
        load(filename, 'experiment');
        UNPACK_STRUCT(experiment, false)
        scores_wo_condition{a} = cell2mat(eval(['score_', acquisition])');
        
        diff_scores{a} = scores_wo_condition{a}-scores{a};
    end
    
    plots =  plot_areaerrorbar_grouped(diff_scores, options);
    maxiter=  size(diff_scores{1},2);
    set(gca, 'Xlim', [1, maxiter])
    plot([1,maxiter], [0,0], '--k', 'Linewidth', linewidth); hold off;
    box off
    ylabel('Value $g(x^*)$');
    title(objective)
end
legend(plots, legends);
legend boxoff
xlabel('Iteration \#')


figname =  'PBO_scores_benchmarks_condition_or_not';
folder = ['C:\Users\tfauvel\Documents\PhD\Figures\Thesis_figures\Chapter_1\',figname];
savefig(fig, [folder,'\', figname, '.fig'])
exportgraphics(fig, [folder,'\' , figname, '.pdf']);
exportgraphics(fig, [folder,'\' , figname, '.png'], 'Resolution', 300);


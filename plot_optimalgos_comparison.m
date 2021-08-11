function fig =  plot_optimalgos_comparison(objectives, objectives_names, acquisition_funs, names, figure_folder,data_dir, figname, nreps, maxiter, rescaling, suffix)

% data_dir =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_duels_data'];
% figure_folder = [pathname,'/Preference_Based_BO/Figures/'];
% figname =  'PBO_scores_benchmarks';

% objectives = {'GP1d', 'forretal08', 'grlee12', 'levy', 'goldpr', 'camel6', 'Ursem_waves'};
% objectives_names = {'GP1d','Forrester (2008)', 'Gramacy and Lee (2012)', 'Levy', 'Goldstein-Price', 'Six Hump Camel', 'Ursem-Waves'};
% acquisition_funs = {'random_acquisition_pref', 'new_DTS','active_sampling', 'MES', 'brochu_EI', 'bivariate_EI', 'random', 'decorrelatedsparring', 'kernelselfsparring'};
% acquisition_funs = {'DTS','random_acquisition_pref','kernelselfsparring','maxvar_challenge', 'bivariate_EI', 'Brochu_EI', 'Thompson_challenge'};
% names = {'DTS','Random', 'KSS', 'MVC', 'Bivariate EI (Nielsen 2015)', 'EI (Brochu 2010)', 'Thompson Challenge',};
% names = {'Duel Thompson Sampling','Random', 'Kernel Self-Sparring', 'Maximum Variance Challenge', 'Bivariate Expected Improvement (Nielsen 2015)', 'Expected Improvement (Brochu 2010)', 'Thompson Challenge'};
% 
 
nobj = numel(objectives);

nacq = numel(acquisition_funs);
graphics_style_paper;

mr = ceil(nobj/2);
mc = 2;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  [1 1 1];
tiledlayout(mr, mc, 'TileSpacing', 'tight', 'padding','compact');
options.handle = fig;
options.alpha = 0.2;
options.error= 'sem';
options.line_width = linewidth/2;
options.semilogy = false;
options.cmap = C;

rng(1);
colors = colororder;
options.colors = colors;
ninit = 5;

for j = 1:nobj
    nexttile
    objective = char(objectives(j));
    
    if rescaling == 0 && strcmp(objective, 'goldpr')
        options.semilogy = true; %true;
    else
        options.semilogy = false;
    end
    clear('score')
    for a = 1:nacq
        acquisition = acquisition_funs{a};
        filename = [data_dir,'/',objective, '_',acquisition, suffix];
        try
        load(filename, 'experiment');
        UNPACK_STRUCT(experiment, false)
        
        
        legends{a}=char(names(a,:));
        n=['a',num2str(a)];
        
        scores{a} = cell2mat(eval(['score_', acquisition])');
    
        catch 
           
            scores{a} = NaN(nreps, maxiter);
        end
    end
%     benchmarks_results{j} = scores;
%     [ranks, average_ranks]= compute_rank(scores, ninit);
    plots =  plot_areaerrorbar_grouped(scores, options);
    box off
    ylabel('Value $g(x^*)$');
    title(objectives_names(j))
    set(gca, 'Fontsize', Fontsize)
end
legend(plots, legends, 'Fontsize', Fontsize);
legend boxoff
xlabel('Iteration \#')

axes1 = gca;
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.53963583333902 0.062422587936693 0.444441641328221 0.17799635257443]);

set(gca, 'Fontsize', Fontsize)


if strcmp(objectives_names(1), 'refractive')
    ylabel('VA (in logMAR)')
    title('')
end
savefig(fig, [figure_folder,'/', figname, '.fig'])
exportgraphics(fig, [figure_folder,'/' , figname, '.pdf']);
exportgraphics(fig, [figure_folder,'/' , figname, '.png'], 'Resolution', 300);




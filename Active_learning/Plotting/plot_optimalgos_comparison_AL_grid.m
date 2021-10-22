function fig = plot_optimalgos_comparison_AL_grid(objectives, objectives_names, acquisition_funs, names, figure_folder,data_dir, figname, nreps, maxiter, rescaling)
 
nobj = numel(objectives);

nacq = numel(acquisition_funs);
graphics_style_paper;

mr = ceil(nobj/2);
mc = 2;
fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  background_color;
tiledlayout(mr, mc, 'TileSpacing', 'tight', 'padding','compact');
options.handle = fig;
options.alpha = 0.2;
options.error= 'sem';
options.line_width = linewidth/2;
options.semilogy = false;
options.cmap = C;
options.colors = colororder;
rng(1);
 
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
        filename = [data_dir,'/',objective, '_',acquisition];
        try
        load(filename, 'experiment');
        UNPACK_STRUCT(experiment, false)
        legends{a}=char(names(a,:));
        n=['a',num2str(a)];
        
        data = cell2mat(eval(['score_c_', acquisition])');
        data = data(1:nreps, 1:maxiter);
        scores{a} = data;
    
        catch 
           
            scores{a} = NaN(nreps, maxiter);
        end
    end
    benchmarks_results{j} = scores;
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

savefig(fig, [figure_folder,'/', figname, '.fig'])
exportgraphics(fig, [figure_folder,'/' , figname, '.pdf']);
exportgraphics(fig, [figure_folder,'/' , figname, '.png'], 'Resolution', 300);




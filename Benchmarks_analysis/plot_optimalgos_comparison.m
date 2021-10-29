function fig =  plot_optimalgos_comparison(objectives, objectives_names, acquisition_funs, names,...
    figure_folder,data_dir, figname, nreps, maxiter, rescaling, suffix, prefix, optim, score_measure, lines, figsize)

nobj = numel(objectives);

nacq = numel(acquisition_funs);
graphics_style_paper;

mr = ceil(nobj/2);
mc = 2;

if ~isempty(figsize)
    fheight = figsize(1);
    fwidth = figsize(2);
    mr = 1;
else
    fheight(2) = 0.7*fheight(2);
end

if mr <4
    fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
else
    fig=figure();
end

fig.Color =  background_color;
tiledlayout(mr, mc, 'TileSpacing', 'tight', 'padding','compact');
options.handle = fig;
options.alpha = 0.2;
options.error= 'sem';
options.line_width = linewidth/2;
options.semilogy = false;
options.cmap = C;
options.xlim= [1,maxiter];
options.x_axis = 1:maxiter;
options.lines = lines;
rng(1);
colors = colororder;
options.colors = colors;
options.lines = lines;
rescaling_table = load('benchmarks_rescaling.mat', 't');
rescaling_table =rescaling_table.t;

options.semilogy = false;
for j = 1:nobj
    nexttile
    objective = char(objectives(j));


    clear('score')
    for a = 1:nacq
        acquisition = acquisition_funs{a};
        filename = [data_dir,'/',prefix, objective, '_',acquisition, suffix];
        load(filename, 'experiment');
        UNPACK_STRUCT(experiment, false)
 
        legends{a}=char(names(a,:));
        n=['a',num2str(a)];

        score = cell2mat(eval([score_measure, '_', acquisition])'); %%%%%%%%

        if strcmp(optim, 'min')
            score= -score;
        elseif strcmp(optim, 'max_proba') && ~strcmp(score_measure, 'score_c')
            score= normcdf(score);
        end

        scores{a} = score;

        if rescaling == 0 && any(reshape(scores{a},1,[])<0)
            options.semilogy = false;
        end
    end
    plots =  plot_areaerrorbar_grouped(scores, options);
    box off

    if mod(j,2)==1
        if strcmp(score_measure, 'score_c')
            ylabel('$P(c = 1|x^*_t)$');
        elseif strcmp(score_measure, 'score_g') || strcmp(score_measure, 'score')
            ylabel('Value $g(x^*_t)$');
        end
    end

    title(objectives_names(j))
    set(gca, 'Fontsize', Fontsize)

    if j == nobj
        legend(plots, legends, 'Fontsize', Fontsize);
        legend boxoff
    end

    if j == nobj || j ==nobj-1
        xlabel('Iteration \#')
    end
end
axes1 = gca;
legend1 = legend(axes1,'show');
legpos = legend1.Position;
set(legend1,...
    'Position',[0.53963583333902 0.062422587936693 legpos(3) legpos(4)]);

set(gca, 'Fontsize', Fontsize)

savefig(fig, [figure_folder,'/', figname, '.fig'])
exportgraphics(fig, [figure_folder,'/' , figname, '.pdf']);
exportgraphics(fig, [figure_folder,'/' , figname, '.png'], 'Resolution', 300);


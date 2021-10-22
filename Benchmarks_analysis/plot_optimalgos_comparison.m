function fig =  plot_optimalgos_comparison(objectives, objectives_names, acquisition_funs, names,...
    figure_folder,data_dir, figname, nreps, maxiter, rescaling, suffix, prefix, optim, score_measure, lines, figsize)

% data_dir =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_duels_data'];
% figure_folder = [pathname,'/Preference_Based_BO/Figures/'];
% figname =  'PBO_scores_benchmarks';

% objectives = {'GP1d', 'forretal08', 'grlee12', 'levy', 'goldpr', 'camel6', 'Ursem_waves'};
% objectives_names = {'GP1d','Forrester (2008)', 'Gramacy and Lee (2012)', 'Levy', 'Goldstein-Price', 'Six Hump Camel', 'Ursem-Waves'};
% acquisition_funs = {'random_acquisition_pref', 'new_DTS','active_sampling', 'MES', 'brochu_EI', 'bivariate_EI', 'random', 'decorrelatedsparring', 'kernelselfsparring'};
% acquisition_funs = {'DTS','random_acquisition_pref','kernelselfsparring','maxvar_challenge', 'bivariate_EI', 'Brochu_EI', 'Thompson_challenge'};
% names = {'DTS','Random', 'KSS', 'MVC', 'Bivariate EI (Nielsen 2015)', 'EI (Brochu 2010)', 'Thompson Challenge',};
% names = {'Duel Thompson Sampling','Random', 'Kernel Self-Sparring', 'Maximally Uncertain Challenge', 'Bivariate Expected Improvement (Nielsen 2015)', 'Expected Improvement (Brochu 2010)', 'Thompson Challenge'};
%

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
mc = 1

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
    
%     if rescaling_table(rescaling_table.Names == objectives_names(j),:).TakeLog == 1
%         options.semilogy = true; %true;
%     else
%         options.semilogy = false;
%     end
%     
%     if strcmp(score_measure, 'score_c') || strcmp(optim, 'max_proba') 
%         options.semilogy = false;
%     end
    
    clear('score')
    for a = 1:nacq
        acquisition = acquisition_funs{a};
        filename = [data_dir,'/',prefix, objective, '_',acquisition, suffix];
%         try
             load(filename, 'experiment');
            UNPACK_STRUCT(experiment, false)
            
           
            
            legends{a}=char(names(a,:));
            n=['a',num2str(a)];
            
            score = cell2mat(eval([score_measure, '_', acquisition])'); %%%%%%%%
            
%             xbest = cell2mat(eval(['xbest_', acquisition])');
%             Sbest =  mean(xbest(1:2:end,:));
%             Cbest =  mean(xbest(1+1:2:end,:));
%             Cbest(end)
%             Sbest(end)
%             
%             
%             xbest = xbest(1:2:end,:);
%             figure()
%             plot(mean(xbest))         

%             xtrain = cell2mat(eval(['xtrain_', acquisition])');
%             xtrain = xtrain(2+1:3:end, :);
%             
%             figure()
%             plot(mean(xtrain))
            
            %         score = cell2mat(eval(['score_c_', acquisition])'); %%%%%%%%
            if strcmp(optim, 'min')
                score= -score;
            elseif strcmp(optim, 'max_proba') && ~strcmp(score_measure, 'score_c')
                score= normcdf(score);
            end
            
            scores{a} = score;
            
            if rescaling == 0 && any(reshape(scores{a},1,[])<0)
                options.semilogy = false;
            end
%         catch
%             scores{a} = NaN(nreps, maxiter);
%         end
    end
    %     benchmarks_results{j} = scores;
    %     [ranks, average_ranks]= compute_rank(scores, ninit);
     plots =  plot_areaerrorbar_grouped(scores, options);
    box off
    
    if mod(j,2)==1
        if strcmp(score_measure, 'score_c')
            ylabel('$P(c = 1|x^*_t)$');
%             set(gca, 'YLim', [0,1])
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
% set(legend1,...
%     'Position',[0.53963583333902 0.062422587936693 0.444441641328221 0.17799635257443]);
set(legend1,...
    'Position',[0.53963583333902 0.062422587936693 legpos(3) legpos(4)]);

set(gca, 'Fontsize', Fontsize)

% name = objectives_names{1};
% if strcmp(name(1:10), 'refractive')
%     ylabel('VA (in logMAR)')
%     title('')
% end
savefig(fig, [figure_folder,'/', figname, '.fig'])
exportgraphics(fig, [figure_folder,'/' , figname, '.pdf']);
exportgraphics(fig, [figure_folder,'/' , figname, '.png'], 'Resolution', 300);



% figure()
% scatter(1:maxiter, mean(scores{1})); hold on;
% scatter(1:maxiter, mean(scores{2})); hold on;


%Preference learning: synthetic experiments

% function preference_learning(randinit, varargin)

add_bo_module;
close all

data_dir =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_PBO_wo_condition/'];

acquisition_funs = {'Dueling_UCB'};

% acquisition_funs = {'DTS'};

%there is a problem with 'value_expected_improvement'
maxiter =  50;%50; %total number of iterations : 200


nreplicates = 20; %20;

nacq = numel(acquisition_funs);


% wbar = waitbar(0,'Computing...');
rescaling = 0;
if rescaling ==0
load('benchmarks_table.mat')
else
load('benchmarks_table_rescaled.mat')
end

objectives = benchmarks_table.fName; %; 'Ursem_waves';'forretal08'; 'camel6';'goldpr'; 'grlee12';'forretal08'};
nobj =numel(objectives);
seeds = 1:nreplicates;
update_period = maxiter+2;
for j = 1:nobj %nobj
    objective = char(objectives(j));
    
    link = @normcdf;
    modeltype = 'exp_prop';
    [g, theta, model] = load_benchmarks(objective, [], benchmarks_table, rescaling);
    close all
    for a =1:nacq
        acquisition_name = acquisition_funs{a};
        acquisition_fun = str2func(acquisition_name);
        clear('xtrain', 'xtrain_norm', 'ctrain', 'score');
        
        filename = [data_dir,objective,'_',acquisition_name];
            for r=1:nreplicates  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                seed  = seeds(r)
                %             waitbar(((a-1)*nreplicates+r)/(nreplicates*nacq),wbar,'Computing...');
                [xtrain{r}, xtrain_norm{r}, ctrain{r}, score{r}] =  PBO_loop_wo_condition(acquisition_fun, seed, maxiter, theta, g, update_period, model);
            end
            clear('experiment')
            fi = ['xtrain_',acquisition_name];
            experiment.(fi) = xtrain;
            fi = ['xtrain_norm_',acquisition_name];
            experiment.(fi) = xtrain_norm;
            fi = ['ctrain_',acquisition_name];
            experiment.(fi) = ctrain;
            fi = ['score_',acquisition_name];
            experiment.(fi) = score;
            
            filename = [data_dir,objective,'_',acquisition_name];
            close all
            save(filename, 'experiment')
        %         save([data_dir,'/synthetic_experiments_data/', objective, '_', acquisition_funs{a}], 'experiment')
    end
end
% cmap = gray(256);
% colo= othercolor('GnBu7');
% colo = jet;
% Fontsize = 14;
% fig=figure(5);
% fig.Name = 'Estimated maximum';
% fig.Color =  background_color;
% for i = 1:nacq_func
%     errorshaded(1:maxiter,toplot(i,:), std_g(i,:), 'Color',  colo(floor(64*i/nacq_func),:),'LineWidth', 1.5, 'Fontsize', 14); hold on
% end
% hold off;
% xlabel('Iteration','Fontsize',Fontsize)
% ylabel('g(x_c)', 'Fontsize',Fontsize)
% box off;
%
% %%
% figure()
% plot(squeeze(mean(experiment.results,2))')
%Preference learning: synthetic experiments

% function preference_learning(randinit, varargin)

add_bo_module;
close all

data_dir =  ['/home/tfauvel/Documents/BO_toolbox/Active_learning/', 'Data_active_learning_preference/'];

acquisition_funs = {'active_sampling_binary'};
acquisition_fun = @active_sampling_binary;
acquisition_name = 'BALD';
maxiter = 50;%total number of iterations

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
conditions = [0,1];
for j = 1:nobj  
    objective = char(objectives(j));
    
    link = @normcdf;
    modeltype = 'exp_prop';
    [g, theta, model] = load_benchmarks(objective, [], benchmarks_table, rescaling);
    close all
    for c = 1:2
        clear('xtrain', 'xtrain_norm', 'ctrain', 'score');
        condition = conditions(c);
        filename = [data_dir,objective,'_',acquisition_name, '_',num2str(condition)];
            for r=1:nreplicates  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                seed  = seeds(r);
                %             waitbar(((a-1)*nreplicates+r)/(nreplicates*nacq),wbar,'Computing...');
                [xtrain{r}, xtrain_norm{r}, ctrain{r}, score{r}] =  AL_preference_loop(acquisition_fun, seed, maxiter, theta, g, update_period, model,condition);       
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
            
            filename = [data_dir,objective,'_',acquisition_name, '_',num2str(condition)];
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
% fig.Color =  [1 1 1];
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
%Preference learning: synthetic experiments

% function preference_learning(randinit, varargin)

add_bo_module;
close all


acquisition_funs = {'bivariate_EI','Dueling_UCB','EIIG','random_acquisition_pref','kernelselfsparring','maxvar_challenge','Brochu_EI', 'Thompson_challenge','DTS'};
% acquisition_funs = {'bivariate_EI'};
% acquisition_funs = {'EIIG'};

maxiter = 80;%100; %total number of iterations : 200
nreplicates = 40; %20;

 
nacq = numel(acquisition_funs);


% wbar = waitbar(0,'Computing...');
rescaling = 1;
if rescaling ==0
    load('benchmarks_table.mat')
    data_dir =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_duels_data_wo_rescaling/'];
else
    load('benchmarks_table_rescaled.mat')
    data_dir =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_duels_data_rescaling/'];
end

%%%%%%
    data_dir =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_duels_data/'];
%%%%%%
objectives = benchmarks_table.fName;


nobj =numel(objectives);
seeds = 1:nreplicates;
update_period = maxiter+2;
more_repets = 0;

for j = 1:nobj
    objective = char(objectives(j));
    
    [g, theta, model] = load_benchmarks(objective, [], benchmarks_table, rescaling);
    model.link = @normcdf;
    
    model.max_x = [model.ub;model.ub];
    model.min_x = [model.lb;model.lb];
    model.type = 'preference';
    close all
    for a =1:nacq
        acquisition_name = acquisition_funs{a};
        if strcmp(acquisition_name, 'PKG')
            modeltype = 'laplace';
        else
            modeltype = 'exp_prop';
        end
        model.modeltype = modeltype;
        acquisition_fun = str2func(acquisition_name);
        clear('xtrain', 'xtrain_norm', 'ctrain', 'score');
        
        filename = [data_dir,objective,'_',acquisition_name];
        
        if more_repets
            load(filename, 'experiment')
            n = numel(experiment.(['xtrain_',acquisition_name]));
            for k = 1:nrepets
                
                disp(['Repetition : ', num2str(n+k)])
                seed =n+k;
                [experiment.(['xtrain_',acquisition_name]){n+k}, experiment.(['xtrain_norm_',acquisition_name]){n+k}, experiment.(['ctrain_',acquisition_name]){n+k}, experiment.(['score_',acquisition_name]){n+k},experiment.(['xbest_',acquisition_name]){n+k}]=  PBO_loop(acquisition_fun, seed, maxiter, theta, g, update_period, model);
            end
            save(filename, 'experiment')
            
        else
            for r=1:nreplicates  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                seed  = seeds(r)
                %             waitbar(((a-1)*nreplicates+r)/(nreplicates*nacq),wbar,'Computing...');
                [xtrain{r}, xtrain_norm{r}, ctrain{r}, score{r}, xbest{r}] =  PBO_loop(acquisition_fun, seed, maxiter, theta, g, update_period, model);
            end
            save_benchmark_results(acquisition_name, xtrain, xtrain_norm, ctrain, score, xbest, g, objective, data_dir)
        end
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

%Preference learning: synthetic experiments

% function preference_learning(randinit, varargin)

add_bo_module;
close all

data_dir =  [pathname,'/Preference_Based_BO/Data/synthetic_exp_PBO_wo_condition/'];

acquisition_funs = {'DTS', 'random_acquisition_pref','kernelselfsparring','maxvar_challenge','Brochu_EI','bivariate_EI', 'Thompson_challenge'};

% acquisition_funs = {'DTS'};

%there is a problem with 'value_expected_improvement'
maxiter = 10;%50; %total number of iterations : 200

%answer = 'max_mu_g'; %the way I report the maximum, either by maximizing the predictive mean of g, either by maximizing the soft-copeland score, EITHER BY maximizing MU_C: which makes more sense (but this depends on the acquisition function).

nreplicates = 1; %20;

nacq = numel(acquisition_funs);


% wbar = waitbar(0,'Computing...');

load('benchmarks_table.mat')
objectives = benchmarks_table.fName; %; 'Ursem_waves';'forretal08'; 'camel6';'goldpr'; 'grlee12';'forretal08'};
nobj =numel(objectives);
seeds = 1:nreplicates;
update_period = maxiter+2;
for j = 1:nobj %nobj
    objective = char(objectives(j));
    
    link = @normcdf;
    modeltype = 'exp_prop';
    [g, theta, lb, ub, lb_norm, ub_norm, theta_lb, theta_ub, kernelfun, kernelname] = load_benchmarks(objective, [], benchmarks_table);
    close all
    for a =1:nacq
        acquisition_name = acquisition_funs{a};
        acquisition_fun = str2func(acquisition_name);
        clear('xtrain', 'xtrain_norm', 'ctrain', 'score');
        
        filename = [data_dir,objective,'_',acquisition_name];
            for r=1:nreplicates  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                seed  = seeds(r)
                %             waitbar(((a-1)*nreplicates+r)/(nreplicates*nacq),wbar,'Computing...');
                [xtrain{r}, xtrain_norm{r}, ctrain{r}, score{r}] =  PBO_loop_wo_condition(acquisition_fun, seed, lb, ub, maxiter, theta, g, update_period, modeltype, theta_lb, theta_ub, kernelname, kernelfun, lb_norm, ub_norm, link);
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
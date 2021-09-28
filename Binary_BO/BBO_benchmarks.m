clear all

add_bo_module;
% close all

data_dir =  [pathname,'/Binary_BO/Data/'];


acquisition_funs = {'EI_Tesch', 'TS_binary','random_acquisition_binary', 'UCB_binary', 'UCB_binary_latent'}; %, 'bivariate_EI_binary'};
acquisition_funs = {'BKG'} %, 'UCB_binary_latent', 'bivariate_EI_binary'};
 
maxiter = 100; %50total number of iterations : 200

nreplicates = 60;

nacq = numel(acquisition_funs);
nbo = 2;

seeds = 1:nreplicates;
update_period = maxiter+2; % do not update the hyperparameters;

rescaling = 1;
if rescaling == 0
    load('benchmarks_table.mat')
else
    load('benchmarks_table_rescaled.mat')
end

% load('benchmarks_table.mat') %%%%%%%%%%%%%%%%%%%%%%

objectives = benchmarks_table.fName;  
nobj =numel(objectives);


link = @normcdf;
 for j = 1:nobj %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    objective = char(objectives(j));
    [g, theta, model] = load_benchmarks(objective, [], benchmarks_table, rescaling);
    close all
    model.ns = 0;
    model.link = link;
    model.type = 'classification';
    model.task = 'max';
    for a =1:nacq
        acquisition_name = acquisition_funs{a};
        if strcmp(acquisition_name, 'BKG')
            modeltype = 'laplace';
        else
            modeltype = 'exp_prop';
        end
        model.modeltype = modeltype;
        
        acquisition_fun = str2func(acquisition_name);
        clear('xtrain', 'xtrain_norm', 'ctrain', 'score');
        for r=1:nreplicates
            seed  = seeds(r)
            [xtrain{r}, xtrain_norm{r}, ctrain{r}, score_c{r}, score_g{r},xbest_c{r}, xbest_g{r}] =  BBO_loop(acquisition_fun, nbo, seed, maxiter, theta, g, update_period, model);
        end
        fi = ['xtrain_',acquisition_name];
        experiment.(fi) = xtrain;
        fi = ['xtrain_norm_',acquisition_name];
        experiment.(fi) = xtrain_norm;
        fi = ['ctrain_',acquisition_name];
        experiment.(fi) = ctrain;
        
        fi = ['score_c_',acquisition_name];
        experiment.(fi) = score_c;
        fi = ['xbest_c_',acquisition_name];        
        experiment.(fi) = xbest_c;
        
        fi = ['score_g_',acquisition_name];
        experiment.(fi) = score_g;
        fi = ['xbest_g_',acquisition_name];        
        experiment.(fi) = xbest_g;
        
        filename = [data_dir,objective,'_',acquisition_name];
        save(filename, 'experiment')        
    end
 end

 
 
 %%
% for j = 1:nobj %%%%%%%%%%%%%%%%9
%     objective = char(objectives(j));
%     [g, theta, model] = load_benchmarks(objective, [], benchmarks_table, rescaling);
%     close all 
%     for a =1:nacq
%         acquisition_name = acquisition_funs{a};
%         if strcmp(acquisition_name, 'BKG')
%             modeltype = 'laplace';
%         else
%             modeltype = 'exp_prop';
%         end
%         model.modeltype = modeltype;
%         
%         acquisition_fun = str2func(acquisition_name);       
%         
%         filename = [data_dir,objective,'_',acquisition_name,'.mat'];
%         source = [data_dir,'BO_toolbox',objective,'_',acquisition_name,'.mat'];
%         
%         try
%         movefile(source, filename)
%         end
%         %save(filename, 'experiment')
%     end
% end
% 
% 
%   
classdef optimization
    properties
        task % Type of task
        maxiter % Number of iterations
        nopt % 
        ninit %Number of time steps before starting learning hyperparameters
         
        update_period
        identification
        hyps_update
        acquisition_fun
        ns % Context variables
    end
    methods
        function optim = optimization(task,identification, nopt, ninit, update_period, hyps_update, acquisition_fun, ns)
            optim.task = task;
            optim.identification =  identification;
            optim.acquisition_fun = acquisition_fun;
            optim.maxiter = maxiter;
            optim.ninit = ninit;
            optim.ns = ns;
            optim.nopt = nopt;
        end
        
        function [xtrain, ctrain, score, xbest,theta_evo] = BOloop(optim, seed, theta, g, model)
            xtrain = [];
            xtrain_norm = [];
            ctrain = [];
            xbounds = [model.lb(:), model.ub(:)];
            rng(seed)
            new_x_norm = rand_interval(model.lb_norm,model.ub_norm);
            new_x = new_x_norm.*(model.ub - model.lb)+model.lb;
            ninit= optim.maxiter + 2;
            
            theta_evo = zeros(numel(theta.cov), maxiter);
            
            options.method = 'lbfgs';
            ncandidates= 10;
            %% Compute the kernel approximation if needed
            if strcmp(model.kernelname, 'Matern52') || strcmp(model.kernelname, 'Matern32') || strcmp(model.kernelname, 'ARD')
                approximation.method = 'RRGP';
            else
                approximation.method = 'SSGP';
            end
            approximation.decoupled_bases = 1;
            approximation.nfeatures = 4096;
            [approximation.phi, approximation.dphi_dx] = sample_features_GP(theta(:), model, approximation);
            x_best_norm_c = zeros(D, optim.maxiter);
            x_best_c = zeros(D, optim.maxiter);
            x_best_norm_g = zeros(D, optim.maxiter);
            x_best_g = zeros(D, optim.maxiter);
            
            score = zeros(1,optim.maxiter);
            
            
            for i =1:optim.maxiter
                disp(i)
                new_c = model.link(g(new_x))>rand;
                xtrain = [xtrain, new_x];
                xtrain_norm = [xtrain_norm, new_x_norm];
                ctrain = [ctrain, new_c];
                
                
                if i > ninit
                    %Local optimization of hyperparameters
                    if mod(i, update_period) ==0
                        theta = model_selection(model, xtrain_norm, ytrain, theta, optimization.update);
                        [approximation.phi, approximation.dphi_dx] = sample_features_GP(theta(:), model, approximation);
                        
                    end
                end
                post =  model.prediction(theta, xtrain_norm, ctrain, [], model, []);
                
                if i> nopt
                    [new_x, new_x_norm] = optim.acquisition_fun(theta, xtrain_norm, ctrain,model, post, approximation);
                else
                    new_x_norm = rand_interval(model.lb_norm,model.ub_norm);
                    new_x = new_x_norm.*(model.ub - model.lb)+model.lb;
                end
                init_guess = [];
                
                if strcmp(optim.identification, 'mu_c')
                    x_best_norm_c(:,i) = model.maxproba(theta, xtrain_norm, ctrain, post);
                    x_best_c(:,i) = x_best_norm_c(:,i) .*(model.ub-model.lb) + model.lb;
                    score_c(i) = normcdf(g(x_best_c(:,i)));
                elseif strcmp(optim.identification, 'mu_g')
                    x_best_norm_g(:,i) = model.maxmean(theta, xtrain_norm, ctrain, post);
                    x_best_g(:,i) = x_best_norm_g(:,i) .*(model.ub-model.lb) + model.lb;
                    score_g(i) = g(x_best_g(:,i));
                elseif strcmp(optim.identification, 'training_set')
                    mu_ytrain =  model.prediction(theta, xtrain_norm, ytrain, xtrain_norm, post);
                    [max_ytrain,b]= max(mu_ytrain);
                end
                theta_evo(:, i) = theta.cov;                
            end
        end
    end
end


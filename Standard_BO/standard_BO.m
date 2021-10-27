classdef standard_BO < optimization
    properties
        noise
    end
    methods
        function optim = standard_BO(objective, task,identification, maxiter, nopt, ninit, update_period, hyps_update, acquisition_fun, ns, noise)
            optim = optim@optimization(objective, task,identification, maxiter, nopt, ninit, update_period, hyps_update, acquisition_fun, ns)
            optim.noise = noise;
        end

        function new_c = query(optim, new_x)
            new_c = optim.objective(new_x) + optim.noise*randn(1, size(new_x,2));
        end
        function score = eval(optim, x_best)
            score = optim.objective(x_best);
        end

        function  [xbest_norm, xbest] = identify(optim, model, theta, xtrain_norm, ctrain, post)
            if strcmp(optim.identification, 'mu_g')
                xbest_norm = model.maxmean(theta, xtrain_norm, ctrain, post);
            elseif strcmp(optim.identification, 'training_set')
                mu_ytrain =  model.prediction(theta, xtrain_norm, ytrain, xtrain_norm, post);
                [max_ytrain,b]= max(mu_ytrain);
                xbest_norm = xtrain_norm(:,b);
            end
            xbest = xbest_norm.*(model.ub-model.lb) + model.lb;
        end       

        function [new_x, new_x_norm] = optim.random_scheme(model)           
            new_x_norm = rand_interval(model.lb_norm,model.ub_norm);
            new_x = new_x_norm.*(model.ub - model.lb)+model.lb;
        end
    end
end


classdef binary_BO < optimization
    methods
        function optim = binary_BO(objective, task,identification, maxiter, nopt, ninit, update_period, hyps_update, acquisition_fun, ns)
            optim = optim@optimization(objective, task,identification, maxiter, nopt, ninit, update_period, hyps_update, acquisition_fun, ns)
        end

        function new_c = query(optim, new_x)
            new_c = normcdf(optim.objective(new_x))>rand(1, size(new_x,2));
        end
        function score = eval(optim, x_best)
            score = normcdf(optim.objective(x_best));
        end

        function  [xbest_norm, xbest] = identify(optim, model, theta, xtrain_norm, ctrain, post)
            xbest_norm = model.maxproba(theta, xtrain_norm, ctrain, post);
            xbest = xbest_norm.*(model.ub-model.lb) + model.lb;
        end

        function [new_x, new_x_norm] = random_scheme(optim, model)   

            if optim.grid
                new_x = optim.
            else
                new_x_norm = rand_interval(model.lb_norm,model.ub_norm);                 
            end
            new_x = new_x_norm.*(model.ub - model.lb)+model.lb;
        end
    end
end


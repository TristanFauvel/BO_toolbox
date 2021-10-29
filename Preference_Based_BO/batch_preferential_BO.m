classdef batch_preferential_BO < optimization
    methods
        function optim = batch_preferential_BO(objective, task,identification, maxiter, nopt, ninit, update_period, hyps_update, acquisition_fun, ns, batch_size)
            optim = optim@optimization(objective, task,identification, maxiter, nopt, ninit, update_period, hyps_update, acquisition_fun, ns)
            optim.tsize = batch_size;
        end

        function new_c = query(optim, new_x)
            gmat = optim.objective(new_x);
            iduels = nchoosek(1:optim.tsize,2)';
            gmat = gmat(iduels);
            new_c= normcdf(gmat(1,:) - gmat(2,:)) > rand(1, size(iduels,2));
        end
        function score = eval(optim, x_best)
            score = optim.objective(x_best);
        end

        function  [xbest_norm, xbest] = identify(optim, model, theta, xtrain_norm, ctrain, post)
            xbest_norm = model.maxmean(theta, xtrain_norm, ctrain, post);
            xbest = xbest_norm.*(model.ub-model.lb) + model.lb;
        end

        function [new_x, new_x_norm] = random_scheme(optim, model)
            samples = rand_interval(model.lb_norm,model.ub_norm, 'nsamples', 2);
            x_duel1 = samples(:,1);
            x_duel2 = samples(:,2);
            new_x_norm= [x_duel1; x_duel2];


            new_x = new_x_norm.*([model.ub;model.ub] - [model.lb; model.lb])+[model.lb; model.lb];

        end
    end
end


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
        objective
    end
    methods
        function optim = optimization(objective, task,identification, maxiter, nopt, ninit, update_period, hyps_update, acquisition_fun, ns)
            optim.task = task;
            optim.identification =  identification;
            optim.acquisition_fun = acquisition_fun;
            optim.maxiter = maxiter;
            optim.ninit = ninit;
            optim.ns = ns;
            optim.nopt = nopt;
            optim.update_period = update_period;
            optim.hyps_update = hyps_update;
            optim.objective = objective;
        end

        function [xtrain, ctrain, score, xbest, theta_evo] = optimization_loop(optim, seed, theta, model)
            xtrain = [];
            xtrain_norm = [];
            ctrain = [];
            rng(seed)
            [new_x, new_x_norm] = optim.random_scheme(model);


            theta_evo = zeros(numel(theta.cov), optim.maxiter);

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

            if strcmp(model.type, 'preference')
                [approximation.phi_pref, approximation.dphi_pref_dx, approximation.phi, approximation.dphi_dx]= sample_features_preference_GP(theta, model, approximation);
            else
                [approximation.phi, approximation.dphi_dx] = sample_features_GP(theta, model, approximation);
            end

            xbest_norm = zeros(model.D, optim.maxiter);
            xbest = zeros(model.D, optim.maxiter);
            score = zeros(1,optim.maxiter);

            for i =1:optim.maxiter
                disp(i)
                new_c = optim.query(new_x);
                xtrain = [xtrain, new_x];
                xtrain_norm = [xtrain_norm, new_x_norm];
                ctrain = [ctrain, new_c];


                if i > optim.ninit
                    %Local optimization of hyperparameters
                    if mod(i, optim.update_period) ==0
                        theta = model.model_selection(xtrain_norm, ctrain, theta, optim. hyps_update);
                        if strcmp(model.type, 'preference')
                            [approximation.phi_pref, approximation.dphi_pref_dx, approximation.phi, approximation.dphi_dx]= sample_features_preference_GP(theta, model, approximation);
                        else
                            [approximation.phi, approximation.dphi_dx] = sample_features_GP(theta, model, approximation);
                        end
                    end
                end
                post =  model.prediction(theta, xtrain_norm, ctrain, [], []);

                if i> optim.nopt
                    [new_x, new_x_norm] = optim.acquisition_fun(theta, xtrain_norm, ctrain,model, post, approximation);
                else
                    [new_x, new_x_norm] = optim.random_scheme(model);
                end

                [xbest_norm(:,i),xbest(:,i)] = optim.identify(model, theta, xtrain_norm, ctrain, post);
                model.xbest_norm = xbest_norm(:,i);
                score(i) = optim.eval(xbest(:,i));

                theta_evo(:, i) = theta.cov;
            end
        end
    end
end


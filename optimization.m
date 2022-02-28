classdef optimization
     properties (Constant)
          AF_ncandidates = 10;
     end
     
    properties
        task % Type of task
        maxiter % Number of iterations
        nopt %
        ninit %Number of time steps before starting learning hyperparameters
        update_period
        identification
        hyps_update
        acquisition_fun
        objective
        batch_size = 1; %batch size, default is 1 (one query at each iteration)
        grid = false;
        D
        ns = 0;
%         xdims
%         sdims
%         s0
    end
    methods
        function optim = optimization(objective, task,identification, maxiter, nopt, ninit, update_period, hyps_update, acquisition_fun, D, ns)
            optim.task = task;
            optim.identification =  identification;
            optim.acquisition_fun = acquisition_fun;
            optim.maxiter = maxiter;
            optim.ninit = ninit;
            optim.nopt = nopt;
            optim.update_period = update_period;
            optim.hyps_update = hyps_update;
            optim.objective = objective;
            optim.D = D;
            optim.ns = ns;
        end

        function [xtrain, ctrain, score, xbest, theta_evo] = optimization_loop(optim, seed, theta, model)
            xtrain = [];
            xtrain_norm = [];
            ctrain = [];
            rng(seed)
            [new_x, new_x_norm] = optim.random_scheme(model);

            theta_evo = zeros(numel(theta.cov), optim.maxiter);

            %% Compute the kernel approximation if needed
            if strcmp(model.kernelname, 'Matern52') || strcmp(model.kernelname, 'Matern32') || strcmp(model.kernelname, 'ARD')
                approximation.method = 'RRGP';
            else
                approximation.method = 'SSGP';
            end
            approximation.decoupled_bases = 1;
            approximation.nfeatures = 4096;

            model = approximate_kernel(model, theta, approximation);
%             if strcmp(model.type, 'preference')
%                 [approximation.phi_pref, approximation.dphi_pref_dx, approximation.phi, approximation.dphi_dx]= sample_features_preference_GP(theta, model, approximation);
%
%
%                 %% Change this in future releases
%                 if optim.context
%                     approximation.phi_pref = @(x) x(1,:)'.*kphi_pref(x(2:end,:)); % ntest x nfeatures
%                     approximation.dphi_pref_dx = @(x) [kphi_pref(x(2:end,:))',x(1,:)'.*dkphi_pref_dx(x(2:end,:))]; % nfeatures x D+1
%
%                     approximation.phi = @(x) x(1,:)'.*kphi(x(2:end,:)); %
%                     approximation.dphi_dx = @(x) [kphi(x(2:end,:))',x(1,:)'.*dkphi_dx(x(2:end,:))]; % nfeatures x D+1
%
%
%
%
%                 else
%                 [approximation.phi, approximation.dphi_dx] = sample_features_GP(theta, model, approximation);

                %% Change this in future releases
%                 if optim.context
%                 if strcmp(model.task, 'max')
%                     approximation.phi = @(x) x(1,:)'.*kphi(x(2:end,:)); %
%                     approximation.dphi_dx = @(x) [kphi(x(2:end,:))',x(1,:)'.*dkphi_dx(x(2:end,:))]; % nfeatures x D+1
%                 else
%                     approximation.phi = kphi; %
%                     approximation.dphi_dx = dkphi_dx; % nfeatures x D+1
%                 end
%                 end
                %%


             xbest_norm = zeros(model.D- model.ns, optim.maxiter);
            xbest = zeros(model.D- model.ns, optim.maxiter);
            score = zeros(1,optim.maxiter);

            for i =1:optim.maxiter
                disp(i)
                new_c = optim.query(new_x);

                if optim.batch_size>1
                    if ~strcmp(model.type, 'preference')
                        error('Batch only available for PBO')
                    end
                    ids = nchoosek(1:optim.batch_size,2)';
                    n = size(ids,2);
                    new_x = reshape(new_x(:,ids(:)),2*model.D, n);
                    new_x_norm = reshape(new_x_norm(:,ids(:)),2*model.D, size(ids,2));
                end
                xtrain = [xtrain, new_x];
                xtrain_norm = [xtrain_norm, new_x_norm];
                ctrain = [ctrain, new_c];


                if i > optim.ninit
                    %Local optimization of hyperparameters
                    if mod(i, optim.update_period) ==0
                        theta = model.model_selection(xtrain_norm, ctrain, theta, optim. hyps_update);
%                         if strcmp(model.type, 'preference')
%                             [approximation.phi_pref, approximation.dphi_pref_dx, approximation.phi, approximation.dphi_dx]= sample_features_preference_GP(theta, model, approximation);
%                         else
%                             [approximation.phi, approximation.dphi_dx] = sample_features_GP(theta, model, approximation);
%                         end
                        model = approximate_kernel(model, theta, approximation);
                    end
                end
                post =  model.prediction(theta, xtrain_norm, ctrain, [], []);

                if i> optim.nopt
                    [new_x, new_x_norm] = optim.acquisition_fun(theta, xtrain_norm, ctrain,model, post, approximation, optim);
                else
                    [new_x, new_x_norm] = optim.random_scheme(model);
                end

                [xbest_norm(:,i),xbest(:,i)] = optim.identify(model, theta, xtrain_norm, ctrain, post);
                model.xbest_norm = xbest_norm(:,i);
                score(i) = optim.eval(xbest(:,i), model);
                theta_evo(:, i) = theta.cov;
            end
        end
    end
end

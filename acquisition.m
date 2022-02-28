classdef acquisition
    properties
        method = 'lbfgs'
        ncandidates = optim.AF_ncandidates
        initial_guess
        utility
        dims = [];
    end
    methods
        function AF = define_AF()

        end

        function [new_x,new_x_norm, L]  = acquire(AF)
            [new_x_norm, L] = AF.optimize_AF(@(x) utility(theta, xtrain_norm, x, ctrain, model, post, e), model.lb_norm, model.ub_norm);
           
            new_x = new_x_norm.*(model.ub-model.lb) + model.lb;

        end

        function [best_x, best_val] = optimize_AF(AF, utility, lb, ub)
            % To optimize acquisition functions
           
            %% Multistart seach with minFunc
            best_x= [];

            x0 = lb;
            f = @(x) subfun(x, utility,dims, x0, objective);
            if ~isempty(dims)
                lb = lb(dims);
                ub = ub(dims);
            end

            % Batch initialization for multi-start optimization : select starting points using the method by
            % Balandat et al 2020 (appendix F).

            D = size(lb,1);
            p = haltonset(D,'Skip',1e3,'Leap',1e2);
            p = scramble(p,'RR2');
            q = qrandstream(p);
            n0 = 500;
            sampSize = 1;
            X = zeros(D, n0);
            v = zeros(1,n0);
            for i = 1:n0
                X(:,i) = qrand(q,sampSize);
                v(i) = utility(X(:,i));
            end

            temperature =  1;
            p  = exp(temperature*zscore(v));
            p = p/sum(p);
            starting_points = X(:,randsample(n0, ncandidates, true, p));

            %starting_points = rand_interval(lb, ub, 'nsamples', ncandidates);
            if ~isempty(init_guess)
                starting_points(:,1)= init_guess;
            end

            best_val=inf;
            options.verbose = 0;
            x = [];
            for k = 1:ncandidates
                try
                    x = minConf_TMP(@(x)utility(x), starting_points(:,k), lb(:), ub(:), options);
                    if any(isnan(x(:)))
                        error('x is NaN')
                    end
                    val = utility(x);
                    if val < best_val
                        best_x = x;
                        best_val = val;
                    end
                catch e %e is an MException struct
                    fprintf(1,e.message);
                end
            end

            if isempty(x)
                error('Optimization failed')
            end
            best_val = - best_val;
        end

        function [f, df] = subfun(x, fun,dims, x0)

            if ~isempty(dims)
                x0(dims) = x;
                [f, df] = fun(x0);
                df = df(:,dims);
            else
                [f, df] = fun(x);
            end

          
                f = -f;
                df = -df;
            
        end
    end
end
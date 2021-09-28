function [x_duel1, x_duel2, new_duel] = bivariate_EI(theta, xtrain_norm, ctrain, model, post, approximation)
%'Bivariate EI only possible with duels, not tournaments'
% Bivariate Expected Improvement, as proposed by Nielsen (2015)
%% Find the maximum of the value function
options.method = 'lbfgs';
options.verbose = 1;

D = size(xtrain_norm,1)/2;
n = size(xtrain_norm,2);
ncandidates =model.ncandidates;
init_guess = [];

% x_duel1 = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm, model.ub_norm, ncandidates,init_guess, options);
x = [xtrain_norm(1:D,:), xtrain_norm((D+1):end,:)];

[g_mu_c,  g_mu_y] = prediction_bin(theta, xtrain_norm, ctrain, [x;model.condition.x0*ones(1,2*n)], model, post);
[a,b]= max(g_mu_y);
x_duel1 = x(:,b);

x_duel2 = multistart_minConf(@(x)compute_bivariate_expected_improvement(theta, xtrain_norm, x, ctrain, model, x_duel1, post), model.lb_norm, model.ub_norm, ncandidates,init_guess, options);
x_duel1 = x_duel1.*(model.max_x(1:D)-model.min_x(1:D)) + model.min_x(1:D);
x_duel2 = x_duel2.*(model.max_x((D+1):2*D)-model.min_x((D+1):2*D)) + model.min_x((D+1):2*D);

new_duel = [x_duel1;x_duel2];

end


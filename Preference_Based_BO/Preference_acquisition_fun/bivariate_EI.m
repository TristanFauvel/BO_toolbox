function   [new_x, new_x_norm] = bivariate_EI(theta, xtrain_norm, ctrain, model, post, approximation)
%'Bivariate EI only possible with duels, not tournaments'
% Bivariate Expected Improvement, as proposed by Nielsen (2015)
%% Find the maximum of the value function
options.method = 'lbfgs';
options.verbose = 1;

D = size(xtrain_norm,1)/2;
n = size(xtrain_norm,2);
ncandidates = 10;
init_guess = [];

x = [xtrain_norm(1:D,:), xtrain_norm((D+1):end,:)];

[g_mu_c,  g_mu_y] = model.prediction(theta, xtrain_norm, ctrain, [x;model.condition.x0*ones(1,2*n)], post);
[a,b]= max(g_mu_y);
x_duel1_norm = x(:,b);

x_duel2_norm = multistart_minConf(@(x)compute_bivariate_expected_improvement(theta, xtrain_norm, x, ctrain, model, x_duel1_norm, post), model.lb_norm, model.ub_norm, ncandidates,init_guess, options);
new_x_norm = [x_duel1_norm;x_duel2_norm];
new_x = new_x_norm.*([model.ub;model.ub] - [model.lb; model.lb])+[model.lb; model.lb];

end


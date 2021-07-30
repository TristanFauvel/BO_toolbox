function [x_duel1, x_duel2, new_duel] = Dueling_UCB(theta, xtrain_norm, ctrain, kernelfun, base_kernelfun, modeltype, max_x, min_x, lb_norm, ub_norm, condition, post, ~)
% Dueling UCB, (Benavoli 2020)

D = size(xtrain_norm,1)/2;
%% Find the maximum of the value function
options.method = 'lbfgs';
regularization = 'nugget';

ncandidates= 5;
init_guess = [];
x_duel1 = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, kernelfun, condition.x0,modeltype, post), lb_norm, ub_norm, ncandidates, init_guess, options);

init_guess = [];
x_duel2 = multistart_minConf(@(x)dUCB(theta, xtrain_norm, x, ctrain, kernelfun, x_duel1, modeltype, post,regularization), lb_norm, ub_norm, ncandidates, init_guess, options);

x_duel1 = x_duel1.*(max_x(1:D)-min_x(1:D)) + min_x(1:D);
x_duel2 = x_duel2.*(max_x(D+1:end)-min_x(D+1:end)) + min_x(D+1:end);

new_duel = [x_duel1;x_duel2];

end

function [ucb_val, ducb_dx]= dUCB(theta, xtrain_norm, x, ctrain, kernelfun, x_duel1, modeltype, post,regularization)
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx] =  prediction_bin(theta, xtrain_norm, ctrain, [x; x_duel1], kernelfun,modeltype, post, regularization);
sigma_y = sqrt(sigma2_y);
dsigma_y_dx = dsigma2y_dx./(2*sigma_y);
D = numel(x);
e = norminv(0.975);
ucb_val = mu_y + e*sigma_y;
ucb_val = -ucb_val;
ducb_dx = -(dmuy_dx(1:D) + e*dsigma_y_dx(1:D));
end
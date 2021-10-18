function [x_duel1, x_duel2, new_duel] = Dueling_UCB(theta, xtrain_norm, ctrain, model, post, approximation)
% Dueling UCB, (Benavoli 2020)

D = model.D;
%% Find the maximum of the value function
options.method = 'lbfgs';

ncandidates =model.ncandidates;
init_guess = [];

if ~isnan(post.x_best_norm)
    x_duel1 = post.x_best_norm;
else
    x_duel1 = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);
end

x_duel2 = multistart_minConf(@(x)dUCB(theta, xtrain_norm, x, ctrain, x_duel1, model, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);

x_duel1 = x_duel1.*(model.max_x(1:D)-model.min_x(1:D)) + model.min_x(1:D);
x_duel2 = x_duel2.*(model.max_x(D+1:2*D)-model.min_x(D+1:2*D)) + model.min_x(D+1:2*D);

new_duel = [x_duel1;x_duel2];

end

function [ucb_val, ducb_dx]= dUCB(theta, xtrain_norm, x, ctrain, x_duel1, model, post)
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx] =  model.prediction(theta, xtrain_norm, ctrain, [x; x_duel1], post);
sigma_y = sqrt(sigma2_y);
dsigma_y_dx = dsigma2y_dx./(2*sigma_y);
D = model.D;
e = norminv(0.975);
ucb_val = mu_y + e*sigma_y;
ucb_val = -ucb_val;
ducb_dx = -(dmuy_dx(1:D) + e*dsigma_y_dx(1:D));
end
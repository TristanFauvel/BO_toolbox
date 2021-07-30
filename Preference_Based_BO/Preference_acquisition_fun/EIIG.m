function [x_duel1, x_duel2, new_duel] = EIIG(theta, xtrain_norm, ctrain, kernelfun, base_kernelfun, modeltype, max_x, min_x, lb_norm, ub_norm, condition, post, ~)
% EIIG, (Benavoli 2020)

D = size(xtrain_norm,1)/2;
n = size(xtrain_norm,2);
%% Find the maximum of the value function
options.method = 'lbfgs';

ncandidates= 5;
regularization = 'nugget';

init_guess = [];
xduel1_norm = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, kernelfun, condition.x0,modeltype, post), lb_norm, ub_norm, ncandidates, init_guess, options);


xduel2_norm = multistart_minConf(@(x)eiig(theta, xtrain_norm, x, ctrain, kernelfun, xduel1_norm, modeltype, post,regularization), lb_norm, ub_norm, ncandidates, init_guess, options);

x_duel1 = xduel1_norm.*(max_x(1:D)-min_x(1:D)) + min_x(1:D);
x_duel2 = xduel2_norm.*(max_x(D+1:end)-min_x(D+1:end)) + min_x(D+1:end);

new_duel = [x_duel1;x_duel2];

end

function [eiig_val, deiig_val_dx]= eiig(theta, xtrain_norm, x, ctrain, kernelfun,xduel1_norm, modeltype, post,regularization)
[I, dIdx]= BALD(theta, xtrain_norm, ctrain, [x;xduel1_norm], kernelfun, modeltype, post);
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx] = prediction_bin(theta, xtrain_norm, ctrain, [x;xduel1_norm], kernelfun, modeltype, post, regularization);
k= 0.5;
D = numel(x);
eiig_val = k*log(mu_c) - I;
deiig_val_dx = k*dmuc_dx./mu_c- dIdx;
deiig_val_dx = deiig_val_dx(1:D);
eiig_val_val = -eiig_val;
deiig_val_dx = -deiig_val_dx;
end
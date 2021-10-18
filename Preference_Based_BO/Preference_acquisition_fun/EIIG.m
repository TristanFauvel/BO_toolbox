function [x_duel1, x_duel2, new_duel] = EIIG(theta, xtrain_norm, ctrain, model, post, approximation)
% EIIG, (Benavoli 2020)

D = size(xtrain_norm,1)/2;
n = size(xtrain_norm,2);
%% Find the maximum of the value function
options.method = 'lbfgs';

ncandidates =model.ncandidates;

init_guess = [];

if ~isnan(post.x_best_norm)
    xduel1_norm = post.x_best_norm;
else
    xduel1_norm = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);
end


xduel2_norm = multistart_minConf(@(x)eiig(theta, xtrain_norm, x, ctrain, model, xduel1_norm, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);

x_duel1 = xduel1_norm.*(model.max_x(1:D)-model.min_x(1:D)) + model.min_x(1:D);
x_duel2 = xduel2_norm.*(model.max_x(D+1:2*D)-model.min_x(D+1:2*D)) + model.min_x(D+1:2*D);

new_duel = [x_duel1;x_duel2];

end

function [eiig_val, deiig_val_dx]= eiig(theta, xtrain_norm, x, ctrain, model, xduel1_norm, post)
[I, dIdx]= BALD(theta, xtrain_norm, ctrain, [x;xduel1_norm],model, post);
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx] = model.prediction(theta, xtrain_norm, ctrain, [x;xduel1_norm], post);
k= 0.5;
D = numel(x);
eiig_val = k*log(mu_c) - I;
deiig_val_dx = k*dmuc_dx./mu_c- dIdx;
deiig_val_dx = deiig_val_dx(1:D);
eiig_val = -eiig_val;
deiig_val_dx = -deiig_val_dx;
end
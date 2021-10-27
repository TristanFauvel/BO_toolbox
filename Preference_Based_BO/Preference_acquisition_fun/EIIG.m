function  [new_x, new_x_norm] = EIIG(theta, xtrain_norm, ctrain, model, post, approximation)
% EIIG, (Benavoli 2020)

D = size(xtrain_norm,1)/2;
n = size(xtrain_norm,2);
%% Find the maximum of the value function
options.method = 'lbfgs';

ncandidates = 10;

init_guess = [];
if ~isnan(model.xbest_norm)
    x_duel1_norm = model.xbest_norm;
else
    x_duel1_norm =  model.maxmean(theta, xtrain_norm, ctrain, post);
end

x_duel2_norm = multistart_minConf(@(x)eiig(theta, xtrain_norm, x, ctrain, model, x_duel1_norm, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);


new_x_norm = [x_duel1_norm;x_duel2_norm];
new_x = new_x_norm.*([model.ub;model.ub] - [model.lb; model.lb])+[model.lb; model.lb];

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
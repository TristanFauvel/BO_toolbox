function [x_duel1, x_duel2,new_duel] = MaxEntChallenge(theta, xtrain_norm, ctrain, kernelfun, base_kernelfun, modeltype, max_x, min_x, lb_norm, ub_norm, condition, post, ~)
options.method = 'lbfgs';
options.verbose = 1;
D = size(xtrain_norm,1)/2;
ncandidates =5;
init_guess = [];

x_best_norm = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, kernelfun, condition.x0,modeltype, post), lb_norm, ub_norm, ncandidates,init_guess, options);

x_duel1 =  x_best_norm.*(max_x(1:D)-min_x(1:D)) + min_x(1:D);


x_duel2 = active_sampling_binary(theta, xtrain_norm, ctrain, kernelfun,modeltype, max_x, min_x, [lb_norm;x_best_norm], [ub_norm;x_best_norm], post);
x_duel2 = x_duel2(1:D);
% regularization = 'nugget';
% 
% new = multistart_minConf(@(x)adaptive_sampling(theta, xtrain_norm, ctrain, x, x_best_norm, kernelfun, modeltype, post), lb_norm, ub_norm, ncandidates,init_guess, options);
% x_duel2 = new.*(max_x(1:d)-min_x(1:d)) + min_x(1:d);

new_duel= [x_duel1; x_duel2];
end


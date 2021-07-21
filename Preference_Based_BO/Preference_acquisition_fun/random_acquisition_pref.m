function [x_duel1, x_duel2, new_duel] = random_acquisition_pref(~,~,~,~,~, ~, max_x, min_x, lb_norm, ub_norm, ~, ~, ~)
d = numel(max_x)/2;
x_duel1 = rand_interval(lb_norm,ub_norm);
x_duel2 = rand_interval(lb_norm,ub_norm);
x_duel1 = x_duel1.*(max_x(1:d)-min_x(1:d)) + min_x(1:d);
x_duel2 = x_duel2.*(max_x(1:d)-min_x(1:d)) + min_x(1:d);
new_duel= [x_duel1; x_duel2];

% x_duel1 = rand_acq();
% x_duel2 = rand_acq();
% new_duel= [x_duel1; x_duel2];
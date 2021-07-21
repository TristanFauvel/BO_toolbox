function [new_x, new_x_norm]= random_acquisition_binary(~,~,~,~,~, max_x, min_x, lb_norm, ub_norm, ~, ~)
new_x_norm = rand_interval(lb_norm,ub_norm);
new_x = new_x_norm.*(max_x-min_x) + min_x;


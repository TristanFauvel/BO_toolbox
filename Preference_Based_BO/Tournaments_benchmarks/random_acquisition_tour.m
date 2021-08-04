function x = random_acquisition_tour(~,~,~,~,~, ~, max_x, min_x, lb_norm, ub_norm, ~, ~, ~, tsize)

xnorm= rand_interval(lb_norm,ub_norm,'nsamples', tsize);


x = xnorm.*(max_x-min_x) + min_x;


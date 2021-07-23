function [x, y, theta_init, lb, ub, theta_lb, theta_ub, kernelfun] = load_benchmarks_active_learning_grid(objective, kernelname, lengthscale)

obj = str2func(objective);
obj = obj(lengthscale, kernelname);
g = @(xx) obj.do_eval(xx);
xbounds = obj.xbounds;
D = obj.D;

kernelfun = obj.kernelfun;
theta= obj.theta;
x = obj.x;
y = obj.y;
theta_init = theta;
theta_lb = -10*ones(size(theta_init));
theta_ub = 10*ones(size(theta_init));

lb = xbounds(:,1);
ub = xbounds(:,2);

return

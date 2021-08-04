function new_x = MVT(theta, xtrain_norm, ctrain, kernelfun, base_kernelfun, modeltype, max_x, min_x, lb_norm, ub_norm, condition, post, ~, tsize)
regularization = 'nugget';
options.method = 'lbfgs';
options.verbose = 1;
D = size(xtrain_norm,1)/2;
ncandidates =5;
init_guess = [];

x_best_norm = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, kernelfun, condition.x0,modeltype, post), lb_norm, ub_norm, ncandidates,init_guess, options);


new = multistart_minConf(@(x)pref_var(theta, xtrain_norm, ctrain, x_best_norm , x, kernelfun, modeltype, post, regularization, tsize), repmat(lb_norm,tsize-1,1), repmat(ub_norm,tsize-1,1), ncandidates,init_guess, options);

xnorm = [x_best_norm; new];
xnorm = reshape(xnorm, D, tsize);
new_x = xnorm.*(max_x-min_x) + min_x;
end

function [Vpref , dVpref_dx] = pref_var(theta, xtrain_norm, ctrain, x_best_norm, x, kernelfun, modeltype, post, regularization, tsize)
D = size(x_best_norm,1);
x = [x_best_norm, reshape(x,D, tsize-1)];

iduels = nchoosek(1:tsize,2)';
nduels = size(iduels,2);

xduels = reshape(x(:,iduels(:)), 2*D, nduels);
var_muc = zeros(1,nduels);
dvar_muc_dx = zeros(2*D,nduels);
for i = 1:nduels
[~,~,~,~,~,~,~,~, var_muc(i), dvar_muc_dx(:,i)] =  prediction_bin(theta, xtrain_norm, ctrain,xduels(:,i), kernelfun, modeltype, post, regularization);

end
% [~,~,~,~,~,~,~,~, var_muc, dvar_muc_dx] =  prediction_bin(theta, xtrain_norm, ctrain,xduels, kernelfun, modeltype, post, regularization);


dpref_var_dx = zeros(1,D*tsize); 

rdvar_muc_dx = dvar_muc_dx(:);
id = repmat(iduels(:)',D,1);
id =id(:);
for i = 1:tsize
    dpref_var_dx(1,(i-1)*D+(1:D)) =  sum(reshape(rdvar_muc_dx(id==i),D,[]),2);
end

Vpref= -sum(var_muc);
dVpref_dx = -dpref_var_dx((D+1):end)';
end

% f = @(x) pref_var(theta, xtrain_norm, ctrain, x_best_norm, x, kernelfun, modeltype, post, regularization, tsize)
% N = 100;
% x = rand((tsize-1)*D,N);
% num_res = NaN((tsize-1)*D,N);
% res = NaN((tsize-1)*D,N);
% for i = 1:N
%     num_res(:,i) = squeeze(test_matrix_deriv(f,x(:,i),1e-9));
%     [~, res(:,i)] = pref_var(theta, xtrain_norm, ctrain, x_best_norm, x(:,i), kernelfun, modeltype, post, regularization, tsize);
% end
% figure();
% plot(num_res(:)); hold on ;
% plot(res(:)); hold off;

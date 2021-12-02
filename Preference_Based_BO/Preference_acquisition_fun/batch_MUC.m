function new_x = batch_MUC(theta, xtrain_norm, ctrain, model, post, ~,nsamples)
 options.method = 'lbfgs';
options.verbose = 1;
D = size(xtrain_norm,1)/2;
ncandidates =5;
init_guess = [];

if ~isnan(model.xbest_norm)
    x_best_norm = model.xbest_norm;
else
    x_best_norm =  model.maxmean(theta, xtrain_norm, ctrain, post);
end

new = multistart_minConf(@(x)pref_var(theta, xtrain_norm, ctrain, x_best_norm , x, model, post, nsamples), repmat(model.lb_norm,nsamples-1,1), repmat(model.ub_norm,nsamples-1,1), ncandidates,init_guess, options);

xnorm = [x_best_norm; new];
xnorm = reshape(xnorm, D, nsamples);
new_x = xnorm.*(model.ub-model.lb) + model.lb;
end

function [Vpref , dVpref_dx] = pref_var(theta, xtrain_norm, ctrain, x_best_norm, x, model, post, nsamples)
D = size(x_best_norm,1);
x = [x_best_norm, reshape(x,D,  nsamples-1)];

iduels = nchoosek(1: nsamples,2)';
nduels = size(iduels,2);

xduels = reshape(x(:,iduels(:)), 2*D, nduels);
var_muc = zeros(1,nduels);
dvar_muc_dx = zeros(2*D,nduels);
for i = 1:nduels
[~,~,~,~,~,~,~,~, var_muc(i), dvar_muc_dx(:,i)] =  model.prediction(theta, xtrain_norm, ctrain,xduels(:,i), post);

end
% [~,~,~,~,~,~,~,~, var_muc, dvar_muc_dx] =  model.prediction(theta, xtrain_norm, ctrain,xduels, post);


dpref_var_dx = zeros(1,D*nsamples); 

rdvar_muc_dx = dvar_muc_dx(:);
id = repmat(iduels(:)',D,1);
id =id(:);
for i = 1:nsamples
    dpref_var_dx(1,(i-1)*D+(1:D)) =  sum(reshape(rdvar_muc_dx(id==i),D,[]),2);
end

Vpref= -sum(var_muc);
dVpref_dx = -dpref_var_dx((D+1):end)';
end

% f = @(x) pref_var(theta, xtrain_norm, ctrain, x_best_norm, x, model, post, nsamples)
% N = 100;
% x = rand((nsamples-1)*D,N);
% num_res = NaN((nsamples-1)*D,N);
% res = NaN((nsamples-1)*D,N);
% for i = 1:N
%     num_res(:,i) = squeeze(test_matrix_deriv(f,x(:,i),1e-9));
%     [~, res(:,i)] = pref_var(theta, xtrain_norm, ctrain, x_best_norm, x(:,i), model, post, nsamples);
% end
% figure();
% plot(num_res(:)); hold on ;
% plot(res(:)); hold off;
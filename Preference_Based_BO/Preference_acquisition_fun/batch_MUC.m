function  [new_x, new_x_norm] = batch_MUC(theta, xtrain_norm, ctrain, model, post, ~, optim)

options.method = 'lbfgs';
options.verbose = 1;
ncandidates = optim.AF_ncandidates;
init_guess = [];

if ~isnan(model.xbest_norm)
    x_best_norm = model.xbest_norm;
else
    x_best_norm =  model.maxmean(theta, xtrain_norm, ctrain, post);
end

new =  optimize_AF(@(x)pref_var(theta, xtrain_norm, ctrain, x_best_norm , x, model, post, optim),...
    repmat(model.lb_norm,optim.batch_size-1,1), repmat(model.ub_norm, optim.batch_size - 1,1), ncandidates,init_guess, options);

new_x_norm = [x_best_norm; new];
new_x_norm = reshape(new_x_norm, model.D, optim.batch_size);
new_x = new_x_norm.*(model.ub-model.lb) + model.lb;
end

function [Vpref , dVpref_dx] = pref_var(theta, xtrain_norm, ctrain, x_best_norm, x, model, post, optim)

nsamples = optim.batch_size;
x = [x_best_norm, reshape(x, model.D,  nsamples-1)];

iduels = nchoosek(1: nsamples,2)';
nduels = size(iduels,2);

xduels = reshape(x(:,iduels(:)), 2*model.D, nduels);
var_muc = zeros(1,nduels);
dvar_muc_dx = zeros(2*model.D,nduels);
for i = 1:nduels
    [~,~,~,~,~,~,~,~, var_muc(i), dvar_muc_dx(:,i)] =  model.prediction(theta, xtrain_norm, ctrain,xduels(:,i), post);

end

dpref_var_dx = zeros(1, model.D*nsamples);

rdvar_muc_dx = dvar_muc_dx(:);
id = repmat(iduels(:)',model.D,1);
id =id(:);
for i = 1:nsamples
    dpref_var_dx(1,(i-1)*model.D+(1:model.D)) =  sum(reshape(rdvar_muc_dx(id==i),model.D,[]),2);
end

Vpref= sum(var_muc);
dVpref_dx = dpref_var_dx((model.D+1):end)';
end


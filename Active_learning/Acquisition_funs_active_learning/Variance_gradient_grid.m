function [new_x, new_x_norm, idx, L] = Variance_gradient_grid(x, theta, xtrain_norm, ctrain, kernelfun, modeltype,lb, ub, post)
xog = x;
ngrid = 300;
if size(x,2)>ngrid
    keep = randsample(size(x,2), ngrid);
    x = x(:,keep);
end

xnorm = (x - lb)./(ub-lb);
regularization = 'nugget';
[mu_c,  ~, ~, ~, ~,~,~,~, var_muc] =   prediction_bin(theta, xtrain_norm, ctrain, xnorm, kernelfun, modeltype, post, regularization);

n = size(x,2);
L = zeros(1, n);
for i = 1:n
    L(i) = vargrad(x, theta, xtrain_norm, ctrain, xnorm(:,i), kernelfun, modeltype, mu_c(i), regularization,var_muc(i));
end

idx= find(L==max(L));
if numel(idx)~=1
    idx = randsample(idx,1);
end

new_x = x(:,idx);

new_x_norm = (new_x - lb)./(ub - lb);
idx = find(ismember(xog',new_x', 'rows'));

end
function vargrad_x = vargrad(x,theta, xtrain_norm, ctrain, xnorm, kernelfun, modeltype, mu_c, regularization, var_muc)
c0 = [ctrain, 0];
c1 = [ctrain, 1];
[~,  ~, ~, ~, ~,~,~,~, var_muc0] =  prediction_bin(theta, [xtrain_norm, xnorm], c0, xnorm, kernelfun, modeltype, [], regularization);
[~,  ~, ~, ~, ~,~,~,~, var_muc1] =  prediction_bin(theta, [xtrain_norm, xnorm], c1, xnorm, kernelfun, modeltype, [], regularization);


vargrad_x = var_muc-(mu_c.*var_muc1 + (1-mu_c).*var_muc0);
end

 

% function vargrad_x = vargrad(x,theta, xtrain_norm, ctrain, xnorm, kernelfun, modeltype, mu_c, regularization, var_muc)
% c0 = [ctrain, 0];
% c1 = [ctrain, 1];
% [~,  ~, ~, ~, ~,~,~,~, var_muc0] =  prediction_bin(theta, [xtrain_norm, xnorm], c0, x, kernelfun, modeltype, [], regularization);
% [~,  ~, ~, ~, ~,~,~,~, var_muc1] =  prediction_bin(theta, [xtrain_norm, xnorm], c1, x, kernelfun, modeltype, [], regularization);
% 
% 
% 
% % p = mu_c;
% % figure()
% % plot(var_muc); hold on;
% % plot(var_muc0); hold on;
% % plot(var_muc1); hold on;
% % plot(p.*var_muc1 + (1-p).*var_muc0); hold off;
% % 
% % legend('$V(\Phi(f))$', '$V(\Phi(f)|0)$', '$V(\Phi(f)|1)$', '$E[V(\Phi(f)|c)]$')
% 
% var_muc0 = max(var_muc0);
% var_muc1 = max(var_muc1);
% 
% vargrad_x = max(var_muc)-(mu_c.*var_muc1 + (1-mu_c).*var_muc0);
% end


% 
% c0 = [ctrain, 0];
% c1 = [ctrain,1];
% var_muc0 = zeros(n);
% var_muc1 = zeros(n);
% Evar_muc = zeros(n);
% E_gradvar_muc = zeros(n);
% 
% for i = 1:n
% [~,  ~, ~, ~, ~,~,~,~, var_muc0(i,:)] =  prediction_bin(theta, [xtrain_norm, xnorm(i)], c0, x, kernelfun, modeltype, [], regularization);
% [~,  ~, ~, ~, ~,~,~,~, var_muc1(i,:)] =  prediction_bin(theta, [xtrain_norm, xnorm(i)], c1, x, kernelfun, modeltype, [], regularization);
% E_var_muc(i,:) = (1-mu_c(i))*var_muc0(i,:) + mu_c(i)*var_muc1(i,:) ;
% E_gradvar_muc(i,:) = E_var_muc(i,:) -  var_muc';
% end
% 
% graphics_style_paper;
% figure()
% subplot(1,3,1)
% plot(var_muc)
% pbaspect([1,1,1])
% 
% subplot(1,3,2)
% plot(diag(E_gradvar_muc))
% pbaspect([1,1,1])
% 
% subplot(1,3,3)
% imagesc(reshape(E_gradvar_muc,n,n))
% set(gca, 'Ydir', 'normal', 'fontsize', Fontsize);
% pbaspect([1,1,1])
% colormap(cmap)
% ylabel('x observed')
% ylabel('x')
% colorbar
% 
% figure()
% imagesc(reshape(E_var_muc,n,n))
% set(gca, 'Ydir', 'normal', 'fontsize', Fontsize);
% pbaspect([1,1,1])
% colormap(cmap)
% ylabel('x observed')
% ylabel('x')
% colorbar
% 
% 
% figure()
% scatter(var_muc, diag(E_gradvar_muc))
% 
% 
% figure()
% plot(var_muc); hold on;
% plot(var_muc + L'); hold off;
% 
% 

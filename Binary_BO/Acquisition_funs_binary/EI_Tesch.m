function [new_x, new_x_norm] = EI_Tesch(theta, xtrain_norm, ctrain,model, post, ~)
%Expected improvement criterion by Tesch et al 2013
if ~strcmp(func2str(model.link), 'normcdf')
    error('Function only implemented for a normcdf link')
end

init_guess = [];
options.method = 'lbfgs';
options.verbose = 1;
ncandidates = 5;
[xbest, mu_c_best] = multistart_minConf(@(x)to_maximize_mu_c(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm,  model.ub_norm, ncandidates, init_guess, options);
 mu_c_best = - mu_c_best;
 ybest = norminv(mu_c_best);
new_x_norm = multistart_minConf(@(x)ExpImp(theta, xtrain_norm, ctrain, x, model, post,mu_c_best, ybest), model.lb_norm,  model.ub_norm, ncandidates, init_guess, options);
 new_x = new_x_norm.*(model.ub-model.lb) + model.lb;
end

function [mu_c,  dmuc_dx] = to_maximize_mu_c(theta, xtrain_norm, ctrain, x,model, post)

[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx] =  prediction_bin(theta, xtrain_norm, ctrain, x, model, post);
mu_c = -mu_c;
dmuc_dx= -squeeze(dmuc_dx);

end


function [ei,deidx] = ExpImp(theta, xtrain_norm, ctrain, x, model, post, mu_c_best, ybest)
nsamps= 30000;
[mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx] =  prediction_bin(theta, xtrain_norm, ctrain, x, model, post);

sigma_y = sqrt(sigma2_y);
samples = mu_y + sigma_y.*randn(1,nsamps);
samples(samples<ybest) = [];
ei = mean(model.link(samples) - mu_c_best);
deidx = mean((model.link(samples) - mu_c_best).*(-1./sigma_y+2*(samples-mu_y).*dmuy_dx./sigma2_y+ ...
    ((samples-mu_y)./sigma2_y).^2.*dsigma2y_dx),2);

ei = -ei;
deidx = -deidx;
end

% 
% f = @(x) ExpImp(theta, xtrain_norm, ctrain, x, model, post, mu_c_best, ybest);
% n= 30;
% [p,q] = meshgrid(linspace(0,1,n));
% x= [p(:), q(:)]';
% for i = 1:n^2
%     [ei(i),deidx(:,i)] = ExpImp(theta, xtrain_norm, ctrain, x(:,i), model, post, mu_c_best, ybest);    
%     res(:,i) = test_matrix_deriv(f, x(:,i), 1e-9);
% end
% figure()
% plot(deidx(:)); hold on;
% plot(res(:)); hold off;
% 
% figure(); 
% imagesc(reshape(ei, n ,n));
% 
% figure();
% subplot(1,2,1)
% imagesc(reshape(res(1,:), n ,n));
% subplot(1,2,2)
% imagesc(reshape(res(2,:), n ,n));
% 
% figure();
% subplot(1,2,1)
% imagesc(reshape(deidx(1,:), n ,n));
% subplot(1,2,2)
% imagesc(reshape(deidx(2,:), n ,n));

function [new_x, new_x_norm] = BKG(theta, xtrain_norm, ctrain,model, post, approximation)
if ~strcmp(model.modeltype, 'laplace')
    error('This acquisition function is only implemented with Laplace approximation')
end
init_guess = [];
options.method = 'lbfgs';
options.verbose = 1;
ncandidates = 5;
[xbest, ybest] = multistart_minConf(@(x)to_maximize_mean_bin_GP(theta, xtrain_norm, ctrain, x, model, post), model.lb_norm,  model.ub_norm, ncandidates, init_guess, options);
ybest = -ybest;

c0 = [ctrain(:)', 0];
c1 = [ctrain(:)',1];

new_x_norm  = multistart_minConf(@(x)knowledge_grad(theta, xtrain_norm, ctrain, x,model, post, c0, c1, xbest, ybest,model.lb_norm, model.ub_norm), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);

new_x = new_x_norm.*(model.ub-model.lb) + model.lb;
end

% f = @(x) knowledge_grad(theta, xtrain_norm, ctrain, x,model, post, c0, c1, xbest, ybest,model.lb_norm, model.ub_norm);
% n= 50;
% D = model.D;
% if D ==1
%     xrange= linspace(0,1,n);
%     nsamps= 100;
%     D=1;
%     deidx = zeros(D, n);
%     samps= zeros(D,nsamps);
%     res = zeros(D,n);
%     for i = 1:n
%         [ei(i),deidx(:,i)] = knowledge_grad(theta, xtrain_norm, ctrain, x,model, post, c0, c1, xbest, ybest,model.lb_norm, model.ub_norm);%         for j= 1:nsamps
%             samps(j) = test_matrix_deriv(f, xrange(i), 1e-6);
%        
%         res(:,i) = mean(samps,2);
%     end
%     figure()
%     plot(deidx(:)); hold on;
%     plot(res(:)); hold off;
% 
% elseif D == 2
%     [p,q] = meshgrid(linspace(0,1,n));
%     x= [p(:), q(:)]';
%     nsamps= 1;
%      
%     ei = zeros(1,n^2);
%     deidx = zeros(D, n^2);
%     samps= zeros(D,nsamps);
%     res = zeros(D,n^2);
%     for i = 1:n^2
%         [ei(i),deidx(:,i)] = knowledge_grad(theta, xtrain_norm, ctrain, x(:,i),model, post, c0, c1, xbest, ybest,model.lb_norm, model.ub_norm);
% %         for j= 1:nsamps
% %             samps(:,j) = test_matrix_deriv(f, x(:,i), 1e-8);
% %         end
% %         res(:,i) = mean(samps,2);
%     end
%     figure()
%     plot(deidx(:)); hold on;
%     plot(res(:)); hold off;
% 
%     figure();
%     imagesc(linspace(0,1,n),linspace(0,1,n), reshape(ei, n ,n)); hold on;
%     set(gca, 'Ydir', 'normal');
%     scatter(new_x_norm(1), new_x_norm(2))
%     
% 
%     figure();
%     subplot(1,2,1)
%     imagesc(reshape(res(1,:), n ,n));
%     colorbar
%     subplot(1,2,2)
%     imagesc(reshape(res(2,:), n ,n));
%     colorbar
% 
%     figure();
%     subplot(1,2,1)
%     imagesc(reshape(deidx(1,:), n ,n));
%     colorbar
%     set(gca, 'Ydir', 'normal');
%     subplot(1,2,2)
%     imagesc(reshape(deidx(2,:), n ,n));
%     colorbar
%     set(gca, 'Ydir', 'normal');
%     
%     h = 1/n;%step size
%     dI1= diff(reshape(ei, n ,n), 1, 1)./h;
%     dI2= diff(reshape(ei, n ,n), 1, 2)./h;
%     figure();
%     subplot(1,2,1)
%     imagesc(dI1);
%     colorbar
%     set(gca, 'Ydir', 'normal');
%     subplot(1,2,2)
%     imagesc(dI2);
%     colorbar
%     set(gca, 'Ydir', 'normal');
% end
% 

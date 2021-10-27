function  [new_x, new_x_norm] = Thompson_challenge(theta, xtrain_norm, ctrain, model, post, approximation)
% This function is called Dueling Thompson in Benavoli 2020.
options.method = 'lbfgs';
options.verbose = 1;
D = size(xtrain_norm,1)/2;
ncandidates = 10;
init_guess = [];

if ~isnan(model.xbest_norm)
    x_duel1_norm = model.xbest_norm;
else
    x_duel1_norm =  model.maxmean(theta, xtrain_norm, ctrain, post);
end

loop = 1;
while loop
    loop = 0;
     new = sample_max_preference_GP(approximation, xtrain_norm, ctrain, theta, model, post);

    if all(x_duel1_norm == new)
        loop =1;
    end
end
x_duel2_norm = new;
new_x_norm = [x_duel1_norm;x_duel2_norm];
new_x = new_x_norm.*([model.ub;model.ub] - [model.lb; model.lb])+[model.lb; model.lb];
end

%
%
% %% Analysis of the function in the case where nd = 2
% nsamples = 100;
%
% %% On training data
%
% xtest_norm = [xtrain_norm(1:d,:), xtrain_norm(d+1:end,:)];
% xtest_norm = sort(xtest_norm);
% % xtest = [xtest; x0.*ones(1,size(xtest,2))];
% test= zeros(nsamples,size(xtest_norm,2));
% new = NaN(d,nsamples);
% for k =1:model.nsamples %sample g* from p(g*|D)
%     y = mvnrnd(mu_y, Sigma2_y)';
%     [sample_g, dsample_g_dx] = sample_features_value_GP(theta, xtrain_norm, y, sig2, kernelname, x0, 'method', approximation.method);
%     test(k,:) = sample_g(xtest_norm);
%     x_init =  rand_interval(lb_norm,ub_norm);
%     new(:,k) = minFuncBC(@(x)deriv(x,sample_g, dsample_g_dx), x_init, lb_norm, ub_norm, options); %bounds constraints.
% end
%
% [~,  test_value, test_variance, ~] = model.prediction(theta, xtrain_norm, ctrain, [xtest_norm; x0.*ones(d,size(xtest_norm,2))], kernelfun, kernelname, modeltype, post, regularization);
%
% if d==1
%     figure()
%     errorshaded(xtest_norm,test_value, sqrt(test_variance)); hold on;
%     errorshaded(xtest_norm,mean(test), sqrt(var(test))); hold off;
% elseif d ==2
%     figure()
%     subplot(1,2,1)
%     scatter(xtest_norm(1,:), xtest_norm(2,:), 25, test_value, 'filled')
%     set(gca,'YDir','normal')
%     colorbar
%     pbaspect([1 1 1])
%     subplot(1,2,2)
%     scatter(xtest_norm(1,:), xtest_norm(2,:), 25, mean(test,1), 'filled')
%     set(gca,'YDir','normal')
%     pbaspect([1 1 1])
%     colorbar
%
%     figure()
%     subplot(1,2,1)
%     scatter(xtest_norm(1,:), xtest_norm(2,:), 25, test_variance, 'filled')
%     set(gca,'YDir','normal')
%     colorbar
%     pbaspect([1 1 1])
%     subplot(1,2,2)
%     scatter(xtest_norm(1,:), xtest_norm(2,:), 25, var(test,1), 'filled')
%     set(gca,'YDir','normal')
%     pbaspect([1 1 1])
%     colorbar
% end
%
%
% %% On the array
%
% m=25;
% xrange1 =linspace(lb_norm(1),ub_norm(1),m);
% xrange2 = linspace(lb_norm(2),ub_norm(2),m);
% [p,q] = meshgrid(xrange1,xrange2);
% x_array = [p(:),q(:)]';
%
% [~,  test_value_array, test_variance_array, ~] = model.prediction(theta, xtrain_norm, ctrain, [x_array; x0.*ones(1,size(x_array,2))], kernelfun, kernelname, modeltype, post, regularization);
%
%
% test= zeros(nsamples,size(x_array,2));
% new = NaN(2,nsamples);
% for k =1:model.nsamples %sample g* from p(g*|D)
%     y = mvnrnd(mu_y, Sigma2_y)';
%     [sample_g, dsample_g_dx] = sample_value_GP(theta, xtrain, ctrain, model, approximation, post);
%     test(k,:) = sample_g(x_array);
%     x_init =  rand_interval(lb_norm,ub_norm);
%     new(:,k) = minFuncBC(@(x)deriv(x,sample_g, dsample_g_dx), x_init, lb_norm, ub_norm, options); %bounds constraints.
% end
%
%
% figure()
% subplot(1,2,1)
% imagesc(xrange1, xrange2, reshape(test_value_array, m,m))
% set(gca,'YDir','normal')
% colorbar
% pbaspect([1 1 1])
% subplot(1,2,2)
% imagesc(xrange1, xrange2, reshape(mean(test,1),m,m));
% set(gca,'YDir','normal')
% colorbar
% pbaspect([1 1 1])
%
%
% figure()
% subplot(1,2,1)
% imagesc(xrange1, xrange2, reshape(test_variance_array, m,m))
% set(gca,'YDir','normal')
% colorbar
% pbaspect([1 1 1])
% subplot(1,2,2)
% imagesc(xrange1, xrange2, reshape(var(test,1),m,m))
% pbaspect([1 1 1])
% set(gca,'YDir','normal')
% colorbar
%
%
% %%
% k=0;
% for i = 1:6
%     for j =1:6
%         k=k+1;
%         subplot(6,6,k)
%         imagesc(xrange1, xrange2, reshape(test(k,:),m,m)); hold on;
%         set(gca,'YDir','normal')
%         colorbar
%         pbaspect([1 1 1])
%         scatter(new(1,k),new(2,k), 25, 'k', 'filled'); hold off;
%     end
% end
%
%
%

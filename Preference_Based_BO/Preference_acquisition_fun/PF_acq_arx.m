function [x_duel1, x_duel2] = PF_acq(theta, xtrain_norm, ctrain, kernelfun, base_name, modeltype, max_x, min_x, lb_norm, ub_norm, x0)
d = size(xtrain_norm,1)/2;

MultiObjFnc = 'Kursawe';


MultiObj.nVar = d*2;
MultiObj.var_min = 0.*ones(1,MultiObj.nVar);
MultiObj.var_max = 1.*ones(1,MultiObj.nVar);

params.Np =100*d;        % Population size
params.pc = 0.9;        % Probability of crossover
params.pm = 0.5;        % Probability of mutation
params.maxgen = 500;    % Maximum number of generations
params.ms = 0.05;       % Mutation strength

MultiObj.fun = @(arg)  multiobjective(theta, xtrain_norm, ctrain, arg', kernelfun, base_name, modeltype);
[PFy, PFx] =NSGAII(params,MultiObj); %Minimisation Genetic Algorithm
PFy = -PFy';

idx = randsample(size(PFx,1),1);
new_duel = PFx(idx,:)';
new_duel =new_duel.*(max_x-min_x) + min_x;
x_duel1 = new_duel(1:d);
x_duel2 = new_duel(d+1:end);

end

function output =  multiobjective(theta, xtrain, ctrain, xduel, kernelfun, base_name, modeltype)
[mu_c,  ~, sigma2_y, Sigma2_y] =  prediction_bin(theta,xtrain, ctrain, xduel, kernelfun, base_name, 'modeltype',modeltype);

d= size(xtrain,1)/2;
x0 = 0.5*ones(d,size(xduel,2));
[~, mu_y1] =  prediction_bin(theta,xtrain, ctrain, [xduel(1:d,:); x0], kernelfun, base_name, 'modeltype',modeltype);
[~, mu_y2] =  prediction_bin(theta,xtrain, ctrain,  [xduel(d+1:end,:); x0], kernelfun, base_name, 'modeltype',modeltype);

output = [(mu_c-0.5).^2,-sigma2_y, -mu_y1, -mu_y2]; %minimize (mu_c-0.5)^2, maximize, sigma2_y, mu_y1, mu_y2

end

% N=100;
% x0 = 0.5;
% x = linspace(0,1,N);
% % xduels = [x;x0*ones(1,N)];
% % [mu_c,  mu_y, sigma2_y, Sigma2_y] = prediction_bin(theta, xtrain_norm, ctrain, xduels, kernelfun, base_name, modeltype, post, regularization);
% % Fontsize=14;
% % fig=figure();
% % fig.Name = 'Acquisition function';
% % fig.Color =  [1 1 1];
% % imagesc(x,x,mu_y-mu_y'); hold on;
% % xlabel('x1','Fontsize',Fontsize)
% % ylabel('x2', 'Fontsize',Fontsize)
% % set(gca,'YDir','normal')
% % scatter(PFx(:,1), PFx(:,2), 30,'r','filled');hold off;
% % pbaspect([1 1 1])
% % colorbar
% % 
% % Fontsize=14;
% % fig=figure();
% % fig.Name = 'Acquisition function';
% % fig.Color =  [1 1 1];
% % imagesc(x,x,Sigma2_y); hold on;
% % xlabel('x1','Fontsize',Fontsize)
% % ylabel('x2', 'Fontsize',Fontsize)
% % set(gca,'YDir','normal')
% % scatter(PFx(:,1), PFx(:,2), 30,'r','filled');hold off;
% % pbaspect([1 1 1])
% % colorbar
% 
% 
% [p,q] = ndgrid(x);
% xduels = [p(:),q(:)]';
% [mu_c,  mu_y, sigma2_y, Sigma2_y] = prediction_bin(theta, xtrain_norm, ctrain, xduels, kernelfun, base_name, modeltype, post, regularization);
% 
% Fontsize=14;
% fig=figure();
% fig.Name = '\mu_c';
% fig.Color =  [1 1 1];
% imagesc(x,x,reshape(mu_c,N,N)); hold on;
% xlabel('x1','Fontsize',Fontsize)
% ylabel('x2', 'Fontsize',Fontsize)
% set(gca,'YDir','normal')
% scatter(PFx(:,1), PFx(:,2), 30,'r','filled');hold off;
% pbaspect([1 1 1])
% colorbar
% 
% Fontsize=14;
% fig=figure();
% fig.Name = '\sigma^2_y';
% fig.Color =  [1 1 1];
% imagesc(x,x,reshape(sigma2_y,N,N)); hold on;
% xlabel('x1','Fontsize',Fontsize)
% ylabel('x2', 'Fontsize',Fontsize)
% set(gca,'YDir','normal')
% scatter(PFx(:,1), PFx(:,2), 30,'r','filled');hold off;
% pbaspect([1 1 1])
% colorbar
% 
% [mu_c_acq,  mu_y_acq, sigma2_y_acq] = prediction_bin(theta, xtrain_norm, ctrain, [x; 0.5*ones(size(x))], kernelfun, base_name, modeltype, post, regularization);
% Fontsize=14;
% fig=figure();
% fig.Color =  [1 1 1];
% errorshaded(x, mu_y_acq, sqrt(sigma2_y_acq)); hold on;
% c = linspace(0,255,params.Np);
% scatter(PFx(:,1), zeros(params.Np,1), 30,c,'filled'); hold on
% scatter(PFx(:,2), zeros(params.Np,1), 30,c,'filled'); hold off
% 
% 
% figure();
% plot(x, forretal08(x))
% 
% figure()
% scatter3(PFy(1,:),PFy(2,:),PFy(3,:), 30,'k','filled');
% xlabel('(mu_c-0.5).^2','Fontsize',Fontsize)
% ylabel('-sigma2_y', 'Fontsize',Fontsize)
% zlabel('-mu_y1')
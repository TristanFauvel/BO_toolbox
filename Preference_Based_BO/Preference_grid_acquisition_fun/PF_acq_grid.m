function new_duel = PF_acq_grid(x, theta, xtrain, ctrain, kernelfun, modeltype, m, base_name)

d = size(x,1);

MultiObjFnc = 'Kursawe';


MultiObj.nVar = d*2;
MultiObj.var_min = 0.*ones(1,MultiObj.nVar);
MultiObj.var_max = 1.*ones(1,MultiObj.nVar);

params.Np = 200*d;        % Population size
params.pc = 0.9;        % Probability of crossover
params.pm = 0.5;        % Probability of mutation
params.maxgen = 50;    % Maximum number of generations
params.ms = 0.05;       % Mutation strength

MultiObj.fun = @(arg)  multiobjective(theta, xtrain, ctrain, arg', kernelfun, base_name, modeltype);
[PFy, PFx] =NSGAII(params,MultiObj); %Minimisation Genetic Algorithm
PFy = -PFy';

idx = randsample(size(PFx,1),m);
new_duel = PFx(idx,:)';
end
% new_duel =new_duel.*(max_x-min_x) + min_x(1:d);

function output =  multiobjective(theta, xtrain, ctrain, xduel, kernelfun, base_name, modeltype)
[mu_c,  ~, sigma2_y, Sigma2_y] =  prediction_bin(theta,xtrain, ctrain, xduel, kernelfun, base_name, 'modeltype',modeltype);

d= size(xtrain,1)/2;
x0 = 0.5*ones(d,size(xduel,2));
[~, mu_y1] =  prediction_bin(theta,xtrain, ctrain, [xduel(1:d,:); x0], kernelfun, base_name, 'modeltype',modeltype);
[~, mu_y2] =  prediction_bin(theta,xtrain, ctrain,  [xduel(d+1:end,:); x0], kernelfun, base_name, 'modeltype',modeltype);

 output = [mu_c,-sigma2_y, -mu_y1, -mu_y2]; %minimize mu_c, maximize, sigma2_y, maximize
 
end

% [mu_c_acq,  mu_y_acq, sigma2_y_acq, Sigma2_y_acq] = prediction_bin(theta, xtrain, ctrain, xduels, kernelfun, base_name, modeltype, post, regularization);
% 
% ntest = numel(x);
% Fontsize=14;
% fig=figure();
% fig.Name = 'Acquisition function';
% fig.Color =  [1 1 1];
% imagesc(x,x,reshape(mu_c_acq,ntest,ntest)); hold on;
% xlabel('x1','Fontsize',Fontsize)
% ylabel('x2', 'Fontsize',Fontsize)
% set(gca,'YDir','normal')
% scatter(PFx(:,1), PFx(:,2), 30,'r','filled');hold off;
% pbaspect([1 1 1])
% colorbar
% 
% [mu_c_acq,  mu_y_acq, sigma2_y_acq] = prediction_bin(theta, xtrain, ctrain, [x; 0.5*ones(size(x))], kernelfun, base_name, modeltype, post, regularization);
% Fontsize=14;
% fig=figure();
% fig.Color =  [1 1 1];
% errorshaded(x, mu_y_acq, sqrt(sigma2_y_acq)); hold on;
% scatter(PFx(:), zeros(2*200,1), 30,'k','filled'); hold off

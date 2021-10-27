function new_x = PF_acq_binary(theta, xtrain_norm, ctrain,model, post, approximation)
d = size(xtrain_norm,1)/2;

MultiObjFnc = 'Kursawe';


MultiObj.nVar = d;
MultiObj.var_min = 0.*ones(1,MultiObj.nVar);
MultiObj.var_max = 1.*ones(1,MultiObj.nVar);

params.Np = 200*d;        % Population size
params.pc = 0.9;        % Probability of crossover
params.pm = 0.5;        % Probability of mutation
params.maxgen = 50;    % Maximum number of generations
params.ms = 0.05;       % Mutation strength

MultiObj.fun = @(arg)  multiobjective(theta, xtrain_norm, ctrain, arg',model, 'post', post);
[PFy, PFx] =NSGAII(params,MultiObj); %Minimisation Genetic Algorithm
PFy = -PFy';

idx = randsample(size(PFx,1),1);
new_x = PFx(idx,:)';
new_x =new_duel.*(max_x-min_x) + min_x;

end

function output =  multiobjective(theta, xtrain, ctrain, x,model, post)
regularization = 'nugget';
[mu_c, mu_y, sigma2_y] =  model.prediction(theta,xtrain, ctrain, x, post);

output = [-mu_y, -sigma2_y];
end

% N=100;
% x0 = 0.5;
% x = linspace(0,1,N);
% xduels = [x;x0*ones(1,N)];
% [mu_c,  mu_y, sigma2_y, Sigma2_y] = model.prediction(theta, xtrain_norm, ctrain, xduels, kernelfun, base_name, modeltype, post, regularization);
% 
% Fontsize=14;
% fig=figure();
% fig.Name = 'Acquisition function';
% fig.Color =  background_color;
% imagesc(x,x,mu_y-mu_y'); hold on;
% xlabel('x1','Fontsize',Fontsize)
% ylabel('x2', 'Fontsize',Fontsize)
% set(gca,'YDir','normal')
% scatter(PFx(:,1), PFx(:,2), 30,'r','filled');hold off;
% pbaspect([1 1 1])
% colorbar
% 
% [mu_c_acq,  mu_y_acq, sigma2_y_acq] = model.prediction(theta, xtrain, ctrain, [x; 0.5*ones(size(x))], kernelfun, base_name, modeltype, post, regularization);
% Fontsize=14;
% fig=figure();
% fig.Color =  background_color;
% errorshaded(x, mu_y_acq, sqrt(sigma2_y_acq)); hold on;
% scatter(PFx(:), zeros(2*200,1), 30,'k','filled'); hold off

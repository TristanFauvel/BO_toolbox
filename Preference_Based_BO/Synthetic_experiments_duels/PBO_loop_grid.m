function results = PBO_loop_grid(acquisition_fun, objective, seed, maxiter, ntest)
rng(seed)

modeltype = 'exp_prop'; % Approximation model

base_kernelfun =  @Gaussian_kernelfun_wnoise;%kernel used within the preference learning kernel, for subject = computer
base_kernelname = 'Gaussian_wnoise';

kernelfun = @(theta, xi, xj, training) conditional_preference_kernelfun(theta, base_kernelfun, xi, xj, training);

% negloglike =   str2func('negloglike_bin'); %define the negative log-likelihood function
prediction =   str2func('prediction_bin_preference');%define the prediction (posterior distribution)

link = @normcdf; %inverse link function for the classification model

%% Define the range of inputs
d=2;%dimension of the input space
g = str2func(objective);
if strcmp(objective, 'forretal08')
    x = linspace(0,1, ntest);
    d=1; %dimension of the input space
    theta = [2.6933, 8.6930,-29.2431];
elseif strcmp(objective, 'levy')
    [x1, x2] = ndgrid(linspace(-10,10, ntest));
    x=  [x2(:), x1(:)]';
    theta = [-0.5387, 5.6751,-88.9788];
elseif strcmp(objective, 'goldpr')
    [x1, x2] = ndgrid(linspace(-2,2, ntest));
    x=  [x2(:), x1(:)]';
    theta = [-1 ; 0; -10];
elseif strcmp(objective, 'camel6')
    [x1, x2] = ndgrid(linspace(-3,3, ntest),linspace(-2,2, ntest));
    x=  [x2(:), x1(:)]';
    theta = [ -0.5642, 5.7548,-98.9687];
elseif strcmp(objective, 'GP1d')    
    hyp.cov = [6 ; 1; -15];
    hyp.mean= 0;
    gen_kernelfun = @Gaussian_kernelfun_wnoise;
    d=999; %resolution
    x = linspace(0,1,d);
    g =  mvnrnd(constant_mean(x,0), gen_kernelfun(hyp.cov, x,x, 'false')); %generate a function
    Sigma = exp(hyp.cov(3))*eye(d);
    [~, g] = sample_GP(x, hyp.cov, x, g', Sigma, base_kernelname);    
    theta= hyp.cov;
    d=1;
    x = linspace(0,1,ntest);
end

if d==1
    [A,B]= ndgrid(x);
    xduels = [B(:), A(:)]';
elseif d==2
    [id1, id2] = ndgrid(1:ntest^2, 1:ntest^2);
    id=  [id2(:), id1(:)]';   
    xduels = NaN(4, ntest^4); %all possible duels
    xduels(1:2,:)= x(:,id(1,:)); 
    xduels(3:4,:)= x(:,id(2,:));
end

grange = g(x);

ind = 1:ntest^(d*2);

%% Initialize the experiment
xtrain = [];
ctrain = [];
ninit = 5; % number of time steps before starting using the acquisition function


rng(seed)
new_id = randsample(ind,1);
new_duel= xduels(:,new_id); %initial sample

optim = NaN(1, maxiter);

if strcmp(func2str(acquisition_fun), 'random')
    ninit = maxiter+1;
end

xmu_g_max = NaN(1,maxiter);
xmu_c_max = NaN(1,maxiter);
C= NaN(1,maxiter);
idxmaxC= NaN(1,maxiter);
idxmaxg= NaN(1,maxiter);
idxmaxc= NaN(1,maxiter);

for i =1:maxiter
    
    disp(i)
    x_duel1 = new_duel(1:d,:);
    x_duel2 = new_duel(d+1:end,:);
    %Generate a binary sample
    c = link(g(x_duel1)-g(x_duel2))>rand;
       
    xtrain = [xtrain, new_duel];
    ctrain = [ctrain, c];
    
    
    % Compute the Condorcet Winner
%     [mu_c_acq,  mu_y_acq, sigma2_y_acq, Sigma2_y_acq] = prediction(theta, xtrain, ctrain, xduels, kernelfun, 'modeltype', modeltype);
    [mu_c_acq,  mu_y_acq, sigma2_y_acq, Sigma2_y_acq] = prediction(theta, xtrain, ctrain, xduels, kernelfun, base_kernelname, 'modeltype', modeltype);

    CS= soft_copeland_score(reshape(mu_c_acq, ntest^d, ntest^d));
    [maxC, idxmaxC(i)]= max(CS);
    C(i) =x(:,idxmaxC(i));
    
    mu_g = reshape(mu_y_acq, ntest,ntest);
    mu_g = mu_g(1,:);
    [~, idxmaxg(i)]= max(mu_g);
    xmu_g_max(i) =x(:,idxmaxg(i));
    
    mu_c = reshape(mu_c_acq, ntest,ntest);
    mu_c = mu_c(1,:);
    [~, idxmaxc(i)]= max(mu_c);
    xmu_c_max(i) =x(:,idxmaxc(i));
%     if strcmp(answer,'Copeland_score')
%         [maxC, idxmaxC]= max(C);
%         xbest =x(:,idxmaxC); %Compute the Condorcet winner
%     elseif strcmp(answer,'max_mu_g')
%         %% Another way to return answers:
%         mu_g = reshape(mu_y_acq, ntest,ntest);
%         mu_g = mu_g(1,:);
%         [maxg, idxmaxg]= max(mu_g);
%         xbest =x(:,idxmaxg); %Compute the Condorcet winner
%     end
%     optim(i)= g(xbest);

    
    %Compute the maximum of the value function according to the model
    
    if i>ninit
        %         if i>=nfit && mod(i-ninit,3)==-1 %Global optimization every 10 steps
        %             % Try global optimization to find theta
        %             myobjfun = @(theta)(negloglike(theta, xtrain, ctrain, kernelfun, 'modeltype', modeltype));
        %             gs = GlobalSearch;
        %             gs.Display='iter';
        %             opts = optimoptions(@fmincon,'Algorithm','sqp');
        %             problem = createOptimProblem('fmincon','objective',myobjfun,'x0', theta, 'lb',[-100,-100],'ub',[100,100], 'options', opts);
        %             ms = MultiStart;
        %             theta = run(ms,problem,100);
        %         elseif i>nfit && mod(i-ninit,5)==-1
        %             %% Update the model hyperparameters
        %             theta= minFunc(negloglike, theta, options, xtrain, ctrain, kernelfun, 'modeltype', modeltype);
        %         end
        
        
        %new_duel = acquisition_fun(x, theta, xtrain, ctrain, kernelfun, link, xduels,  mu_y_acq, sigma2_y_acq, Sigma2_y_acq, modeltype, maxC, mu_c_acq, base_kernelname);
        new_duel = acquisition_fun(x, theta, xtrain, ctrain, kernelfun, modeltype, m, base_kernelname);
    else %When we have not started to train the GP classification model, the acquisition is random
        
        new_id = randsample(ind,1);
        new_duel= xduels(:,new_id); %initial sample
        
        %         id_x2 = find(new_duel(2)== x);
        %         id_x1= find(new_duel(1)== x);
    end
end
[gmax, idxmax]= max(g(x));
xmax =x(idxmax);

results.xmax= xmax;
results.gmax=gmax;
results.xtrain = xtrain;
results.ctrain = ctrain;
results.C= C; 
results.xmu_g_max=xmu_g_max;
results.xmu_c_max=xmu_c_max;
results.grange = grange;
results.x = x;
results.idxmaxc = idxmaxc;
results.idxmaxg = idxmaxg;
results.idxmaxC = idxmaxC;
results.theta = theta; 
return

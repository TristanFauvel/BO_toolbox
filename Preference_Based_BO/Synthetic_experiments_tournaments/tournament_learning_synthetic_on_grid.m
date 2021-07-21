function results = tournament_learning_synthetic_on_grid(acquisition_fun, objective, seed, maxiter, ntest, m)
rng(seed)

modeltype = 'exp_prop'; % Approximation model

base_kernelfun =  @Gaussian_kernelfun_wnoise;%kernel used within the preference learning kernel, for subject = computer
%base_kernelfun =  @ARD_kernelfun_wnoise;%kernel used within the preference learning kernel, for subject = computer
kernelname = 'Gaussian_wnoise';
sampling_kernelname = 'Gaussian_wnoise';

kernelfun = @(theta, xi, xj, training) conditional_preference_kernelfun(theta, base_kernelfun, xi, xj, training);

negloglike =   str2func('negloglike_bin'); %define the negative log-likelihood function
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
    xg= linspace(-10,10, ntest);
    [x1g, x2g] = ndgrid(xg);
    x=  [x2g(:), x1g(:)]';
    theta = [-0.5387, 5.6751,-88.9788];
elseif strcmp(objective, 'goldpr')
    xg=linspace(-2,2, ntest);
    [x1g, x2g] = ndgrid(xg);
    x=  [x2g(:), x1g(:)]';
    theta = [-1 ; 0; -10];
elseif strcmp(objective, 'camel6')
    x1g = linspace(-3,3, ntest);
    x2g = linspace(-2,2, ntest);
    [x1g, x2g] = ndgrid(x1g,x2g);
    x=  [x2g(:), x1g(:)]';
    theta = [ -0.5642, 5.7548,-98.9687];
elseif strcmp(objective, 'GP1d')
    hyp.cov = [6 ; 1; -15];
    hyp.mean= 0;
    x = linspace(0,1,ntest);
    g =  mvnrnd(constant_mean(x,0), base_kernelfun(hyp.cov, x,x, false)); %generate a function on a grid
    Sigma2 = exp(-5)*eye(ntest);
    [~, g] = sample_GP(x, hyp.cov, x, g', Sigma2, sampling_kernelname); %continous approximation
    theta= hyp.cov;
    d=1;
elseif strcmp(objective, 'GP2d')
    d=2;
    hyp.cov = [6 ; 1; -15];
    %    hyp.cov = [6; 6 ; 1; -15];
    
    hyp.mean= 0;
    %     gen_kernelfun = @Gaussian_kernelfun_wnoise;
    xg=linspace(0,1, ntest);
    [x1g, x2g] = ndgrid(xg);
    x=  [x2g(:), x1g(:)]';
    g =  mvnrnd(constant_mean(x,0), base_kernelfun(hyp.cov, x,x, false)); %generate a function on a grid
    Sigma2 = exp(-5)*eye(ntest^d);
    [~, g] = sample_GP(x, hyp.cov, x, g', Sigma2, sampling_kernelname); %continous approximation
    
    theta= hyp.cov;
elseif strcmp(objective, 'GPnd')
    d=10;
    hyp.cov = [6 ; 1; -15];   
    hyp.mean= 0;
    xg=linspace(0,1, ntest);
    [x1g, x2g] = ndgrid(xg);
    x=  [x2g(:), x1g(:)]';
    g =  mvnrnd(constant_mean(x,0), base_kernelfun(hyp.cov, x,x, false)); %generate a function on a grid
    Sigma2 = exp(-5)*eye(ntest^d);
    [~, g] = sample_GP(x, hyp.cov, x, g', Sigma2, sampling_kernelname); %continous approximation
    
    theta= hyp.cov;
end

% theta = [ 0, 0,-100]; %%%%%%%%%%%%%%%%%%%%%%%%%%%%
grange = g(x);
ind = ntest^d;

%% Initialize the experiment
xtrain = [];
ctrain = [];
ninit = 15; % number of time steps before starting using the acquisition function

new_id=randsample(ind,m); %sample without replacement
new_duel= x(:, new_id);
new_duel = new_duel(:); %initial sample

optim = NaN(1, maxiter);

if strcmp(func2str(acquisition_fun), 'random')
    ninit = maxiter+1;
end

xmu_g_max = NaN(d,maxiter);
xmu_c_max = NaN(d,maxiter);
C= NaN(1,maxiter);
idxmaxg= NaN(1,maxiter);
idxmaxc= NaN(1,maxiter);

x0 = x(:,1);
for i =1:maxiter
    
    disp(i)
    
    iduels = nchoosek(1:m,2)';
    new_duels =NaN(2*d, size(iduels,2));
    %Convert this into binary comparisons:
    for j = 1:size(iduels, 2)
        k1=iduels(1,j);
        k2=iduels(2,j);
        new_duels(1:d,j)=new_duel((k1-1)*d+1:k1*d);
        new_duels(d+1:end,j)=new_duel((k2-1)*d+1:k2*d);
    end
    % Note : the tournaments are coded as follow: [x1(1); ... ; x1(d) ; ...
    % ; xm(1); ... ; xm(d)]
    % equivalent to : new_duels = nchoosek(new_duel,2)'; for d=1
    
    gvec = g(reshape(new_duel, d, m));
    
    xg = link(gvec-gvec');
    xg(logical(eye(m)))=1;
    weights = prod(xg,2);
    %normalize the weights
    weights =weights/sum(weights);
    
    %Generate a binary sample
    ctour = mnrnd(1,weights); %link(g(x_duel1)-g(x_duel2))>rand;
    
    %Convert this into binary comparisons:
    
    c = nchoosek(ctour,2)';
    idkeep= find(~ismember(c', [0,0], 'rows')); %remove the results that correspond to ties
    
    xtrain = [xtrain, new_duels(:,idkeep)];
    ctrain = [ctrain, c(1,idkeep)];
    
    [mu_c,  mu_g] = prediction(theta, xtrain, ctrain, [x; x0*ones(1,ntest^d)], kernelfun, kernelname, 'modeltype', modeltype);

    
    [~, idxmaxg(i)]= max(mu_g);
    xmu_g_max(:,i) =x(:,idxmaxg(i));
    
    [~, idxmaxc(i)]= max(mu_c);
    xmu_c_max(:,i) =x(:,idxmaxc(i));
    
    %Compute the maximum of the value function according to the model
    
    if i>ninit
        %         if i==ninit+1 %&& mod(i-ninit,3)==-1 %Global optimization every 10 steps
        %             % Try global optimization to find theta
        %             myobjfun = @(theta)(negloglike(theta(:), xtrain, ctrain, kernelfun, 'modeltype', modeltype));
        %             gs = GlobalSearch;
        %             gs.Display='iter';
        %             opts = optimoptions(@fmincon,'Algorithm','sqp');
        %             problem = createOptimProblem('fmincon','objective',myobjfun,'x0', theta, 'lb',[-100,-100],'ub',[100,100], 'options', opts);
        %             ms = MultiStart;
        %             theta = run(ms,problem,100);
        %         else % eif i>nfit && mod(i-ninit,5)==-1
        %             %% Update the model hyperparameters
        %             options=[];
        %             theta= minFunc(negloglike, theta(:), options, xtrain, ctrain, kernelfun, 'modeltype', modeltype);
        %             if theta(2)>10
        %                 disp('stop')
        %             end
        %         end
        %
        %% Update the model hyperparameters
%         options=[];
%         if mod(i-ninit,5)==0
%             theta= minFunc(negloglike, theta(:), options, xtrain, ctrain, kernelfun, 'modeltype', modeltype);
%             
%             if theta(2)>50
%                 disp('stop')
%             end
%         end
%         
       
        new_duel = acquisition_fun(x, theta, xtrain, ctrain, kernelfun,modeltype, m, kernelname);
    else %When we have not started to train the GP classification model, the acquisition is random
        new_id=randsample(1:ind,m); %sample without replacement
        new_duel= x(:, new_id);
        new_duel = new_duel(:);
    end
end
[gmax, idxmax]= max(g(x));
xmax =x(:,idxmax);

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
results.theta = theta;
return

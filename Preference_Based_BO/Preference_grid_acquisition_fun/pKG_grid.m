function [x_duel1, x_duel2,new_duel] = pKG(theta, xtrain_norm, ctrain, kernelfun, base_kernelfun, modeltype, max_x, min_x, lb_norm, ub_norm, condition, post, ~)

% Preference Knowledge Gradient

d = size(xtrain_norm, 1)/2;

if d == 1
    n = 30;
    x = linspace(0,1,n);
    [A,B]= ndgrid(x);
    xduels = [B(:), A(:)]';   
elseif d==2
    n = 15;
        [x1, x2] = ndgrid(linspace(0,1,n));
    x=  [x2(:), x1(:)]';    
    [id1, id2] = ndgrid(1:n^2, 1:n^2);
    id=  [id2(:), id1(:)]';   
    xduels = NaN(4, n^4); %all possible duels
    xduels(1:2,:)= x(:,id(1,:)); 
    xduels(3:4,:)= x(:,id(2,:));    
else 
    error('Preference Knowledge Gradient not implemented for d>2')
    
end

options.method = 'sd';
options.verbose = 1;
d = size(xtrain_norm,1)/2;
ncandidates =5;
init_guess = [];

mu_c = prediction_bin(theta, xtrain_norm, ctrain, xduels, kernelfun,kernelname, modeltype, post, regularization);

x_best_norm = multistart_minConf(@(x)to_maximize_value_function(theta, xtrain_norm, ctrain, x, kernelfun, x0,modeltype, post), lb_norm, ub_norm, ncandidates,init_guess, options);

n= size(x,2);
[~,  gm] = prediction_bin(theta, xtrain_norm, ctrain, [x_best_norm;x0*ones(1,n)], kernelfun,kernelname, modeltype, post, regularization);

KG = zeros(1, size(xduels, 2));
for i = 1:size(xduels,2)
    [~,  mu_y_1, ~] = prediction_bin(theta, [xtrain, xduels(:,i)], [ctrain, 1], [x;x0*ones(1,n)], kernelfun, kernelname, modeltype, post, regularization);
    [~,  mu_y_0, ~] = prediction_bin(theta, [xtrain, xduels(:,i)], [ctrain, 0], [x;x0*ones(1,n)], kernelfun, kernelname, modeltype, post, regularization);

    [maxg1, ~]= max(mu_y_1);  
    [maxg0, ~]= max(mu_y_0);
%     KG(i) = mu_c(i).*(maxg1-gm).*((maxg1-gm)>0)+(1-mu_c(i)).*(maxg0-gm).*((maxg0-gm)>0);
    KG(i) = mu_c(i).*(maxg1-gm)+(1-mu_c(i)).*(maxg0-gm);

end            
            
new_id= find(KG==max(KG));
if numel(new_id)~=1
    new_id = randsample(new_id,1);    
end
new_duel= xduels(:,new_id);
new_duel = new_duel.*(max_x -min_x) + min_x;

x_duel1 = new_duel(1:d,:);
x_duel2 = new_duel(d+1:end,:);
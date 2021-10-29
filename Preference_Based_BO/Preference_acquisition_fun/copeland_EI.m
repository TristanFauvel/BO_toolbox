function  [new_x, new_x_norm] = copeland_expected_improvement(theta, xtrain_norm, ctrain, model, post, ~)

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
    error('Copeland EI not implemented for d>2')
    
end

%(x, theta, xtrain, ctrain, kernelfun, link, xduels,  mu_y_acq, sigma2_y_acq, Sigma2_y_acq, modeltype, C, mu_c_acq)
%C is the value of the Condorcet winner
mu_c = model.prediction(theta, xtrain_norm, ctrain, xduels, post);
C= soft_copeland_score(reshape(mu_c, n, n));
C= max(C);

CEI = zeros(1, size(xduels, 2));
for i = 1:size(xduels,2)
    [mu_c_1, ~, ~] = model.prediction(theta, [xtrain_norm, xduels(:,i)], [ctrain, 1], xduels, post);
    [mu_c_0, ~, ~] = model.prediction(theta, [xtrain_norm, xduels(:,i)], [ctrain, 0], xduels, post);

    C1= soft_copeland_score(reshape(mu_c_1, n, n));
    [maxC1, ~]= max(C1); %value of the condorcet winner in case 1 is returned with the new duel
    C0= soft_copeland_score(reshape(mu_c_0, n, n));  %value of the condorcet winner in case 0 is returned with the new duel
    [maxC0, ~]= max(C0);
    CEI(i) = mu_c(i).*(maxC1-C).*((maxC1-C)>0)+(1-mu_c(i)).*(maxC0-C).*((maxC0-C)>0);
end            
            
new_id= find(CEI==max(CEI));
if numel(new_id)~=1
    new_id = randsample(new_id,1);    
end
new_x_norm= xduels(:,new_id); 
new_x = new_x_norm.*([model.ub;model.ub] - [model.lb; model.lb])+[model.lb; model.lb];


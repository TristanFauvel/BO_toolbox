function new_duel = copeland_expected_improvement_grid(x, theta, xtrain, ctrain, kernelfun, xduels, modeltype, m, kernelname, post)

%(x, theta, xtrain, ctrain, kernelfun, link, xduels,  mu_y_acq, sigma2_y_acq, Sigma2_y_acq, modeltype, C, mu_c_acq)
%C is the value of the Condorcet winner
if m > 2 
error('Copeland EI only possible with duels, not tournaments')
end
n= size(x,2);
CEI = zeros(1, size(xduels, 2));
for i = 1:size(xduels,2)
    [mu_c_1, ~, ~] = model.prediction(theta, [xtrain, xduels(:,i)], [ctrain, 1], xduels, kernelfun,kernelname, modeltype, post, regularization);
    [mu_c_0, ~, ~] = model.prediction(theta, [xtrain, xduels(:,i)], [ctrain, 0], xduels, kernelfun, kernelname,modeltype, post, regularization);

    C1= soft_copeland_score(reshape(mu_c_1, n, n));
    [maxC1, ~]= max(C1); %value of the condorcet winner in case 1 is returned with the new duel
    C0= soft_copeland_score(reshape(mu_c_0, n, n));  %value of the condorcet winner in case 0 is returned with the new duel
    [maxC0, ~]= max(C0);
    CEI(i) = mu_c_acq(i).*(maxC1-C).*((maxC1-C)>0)+(1-mu_c_acq(i)).*(maxC0-C).*((maxC0-C)>0);
end            
            
new_id= find(CEI==max(CEI));
if numel(new_id)~=1
    new_id = randsample(new_id,1);    
end
new_duel= xduels(:,new_id);


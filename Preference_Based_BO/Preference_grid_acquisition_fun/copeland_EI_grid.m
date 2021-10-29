function CEI = copeland_ei(theta, xtrain_norm, ctrain, x, xduels, model, post, C,n)
c0 = [ctrain(:); 0];
c1 = [ctrain(:); 1];
mu_c = model.prediction(theta, xtrain_norm, ctrain, x, post);

[mu_c_1, ~, ~] = model.prediction(theta, [xtrain_norm, x], c1, xduels, []);
[mu_c_0, ~, ~] = model.prediction(theta, [xtrain_norm, x], c0, xduels, []);

C1= soft_copeland_score(reshape(mu_c_1, n, n));
[maxC1, ~]= max(C1); %value of the condorcet winner in case 1 is returned with the new duel
C0= soft_copeland_score(reshape(mu_c_0, n, n));  %value of the condorcet winner in case 0 is returned with the new duel
[maxC0, ~]= max(C0);
CEI  = mu_c.*(maxC1-C).*((maxC1-C)>0)+(1-mu_c).*(maxC0-C).*((maxC0-C)>0);


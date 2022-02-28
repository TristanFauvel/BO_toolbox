function  [new_x, new_x_norm] = MaxEntChallenge(theta, xtrain_norm, ctrain, model, post, approximation, optim)

D = model.D;

if ~isnan(model.xbest_norm)
    x_duel1_norm = model.xbest_norm;
else
    x_duel1_norm =  model.maxmean(theta, xtrain_norm, ctrain, post);
end
x_duel1= x_duel1_norm.*(model.ub- model.lb)+model.lb;
 
model2 = model;
model2.lb_norm = [model.lb_norm;x_duel1_norm];
model2.ub_norm = [model.ub_norm;x_duel1_norm];
model2.lb = [model.lb;x_duel1];
model2.ub = [model.ub;x_duel1];

[x_duel2, x_duel2_norm] = BALD(theta, xtrain_norm, ctrain, model2, post);

x_duel2 = x_duel2(1:D);
x_duel2_norm = x_duel2_norm(1:D);
new_x_norm = [x_duel1_norm;x_duel2_norm];
new_x =  [x_duel1;x_duel2];
end


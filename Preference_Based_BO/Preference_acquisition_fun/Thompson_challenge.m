function  [new_x, new_x_norm] = Thompson_challenge(theta, xtrain_norm, ctrain, model, post, approximation, optim)
% This function is called Dueling Thompson in Benavoli 2020.

if ~isnan(model.xbest_norm)
    x_duel1_norm = model.xbest_norm;
else
    x_duel1_norm =  model.maxmean(theta, xtrain_norm, ctrain, post);
end

loop = 1;
while loop
    loop = 0;
     new = model.sample_max_GP(approximation, xtrain_norm, ctrain, theta, post);

    if all(x_duel1_norm == new)
        loop =1;
    end
end
x_duel2_norm = new;
new_x_norm = [x_duel1_norm;x_duel2_norm];
new_x = new_x_norm.*([model.ub;model.ub] - [model.lb; model.lb])+[model.lb; model.lb];
end



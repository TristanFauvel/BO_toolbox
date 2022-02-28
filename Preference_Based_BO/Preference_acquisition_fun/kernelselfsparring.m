function  [new_x, new_x_norm] = kernelselfsparring(theta, xtrain_norm, ctrain, model, post, approximation, optim)
nsamples = 2;
for k =1:nsamples %sample g* from p(g*|D)
    loop = 1;
    while loop
        loop = 0;
        [new_norm, new] = model.sample_max_GP(approximation, xtrain_norm, ctrain, theta, post);

        if k == 2  && all(x_duel1 == new)
            loop =1;
        end
    end
    if k==1
        x_duel1 = new;
        x_duel1_norm = new_norm;
    elseif k ==2
        x_duel2 = new;
        x_duel2_norm = new_norm;

    end
end
new_x= [x_duel1; x_duel2];
new_x_norm = [x_duel1_norm;x_duel2_norm];

end

function U = knowledge_grad_grid(theta, xtrain_norm, ctrain, xx, xt,model, post, c0, c1, ybest)

mu_c =  prediction_bin(theta, xtrain_norm, ctrain, xt, model, post);

[mu_c1, mu_y1] =  prediction_bin(theta,  [xtrain_norm, xt], c1, xx, model, []);
[mu_c0, mu_y0] =  prediction_bin(theta,  [xtrain_norm, xt], c0, xx, model, []);

[ybest1, b] = max(mu_y1);
 
[ybest0, b] = max(mu_y0);
 
U = mu_c.*ybest1 + (1-mu_c).*ybest0 -ybest;

end
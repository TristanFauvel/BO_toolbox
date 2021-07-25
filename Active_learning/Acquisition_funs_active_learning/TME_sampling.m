function  [new_x, idx, L] = TME_sampling(x, theta, xtrain, ctrain,kernelfun, modeltype)

%% Direct method
nx = size(x,2);
xrange = x ; %linspace(0,1,nx);
gaussian_entropy = @(Sigma) 0.5+log(det(Sigma)) + 0.5*size(Sigma,1)*log(2*pi*exp(1));
post = [];
regularization = 'nugget';
[mu_c,  mu_y, sigma2_y] =  prediction_bin(theta, xtrain, ctrain, xrange, kernelfun, modeltype, post, regularization);

%% Total marginal entropy
H1 = 0;
for i = 1:nx
    H1 = H1 + gaussian_entropy(sigma2_y(i));
end

    post = [];

%% Expected total marginal entropy after observing y
c0 = [ctrain,0];
c1 = [ctrain,1];
H2 = zeros(numel(x),1);
for i =1:numel(x)
    %case y = 0
    [~, ~, sigma2_y0] =  prediction_bin(theta, [xtrain, x(:,i)], c0, xrange, kernelfun, modeltype, post, regularization);
    %case y = 1
    [~,~, sigma2_y1] =  prediction_bin(theta, [xtrain, x(:,i)], c1, xrange, kernelfun, modeltype, post, regularization);
    H20 = 0;
    H21 = 0;
    for j = 1:nx
        H20= H20 +  gaussian_entropy(sigma2_y0(j));
        H21= H21 + gaussian_entropy(sigma2_y1(j));
    end
    H2(i) = mu_c(i)*H21 + (1-mu_c(i))*H20;
end
I= H1-H2;

L = I;
[a,idx] = max(L);
new_x = x(:,idx);
dIdx =[];

return


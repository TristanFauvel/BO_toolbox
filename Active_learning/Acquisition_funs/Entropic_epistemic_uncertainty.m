function [I, dIdx]= Entropic_epistemic_uncertainty(theta, xtrain, ctrain, x, model, post)

modeltype = model.modeltype;

if nargout>1
    [mu_c,  mu_y, sigma2_y, Sigma2_y, dmuc_dx, dmuy_dx, dsigma2y_dx] =  model.prediction(theta, xtrain, ctrain, x, post);
else
    [mu_c,  mu_y, sigma2_y] =  model.prediction(theta, xtrain, ctrain, x, post);
    
end

h = @(p) -p.*log(p+eps) - (1-p).*log(1-p+eps);


if strcmp(modeltype, 'exp_prop')
    % for a gaussian cdf link function:
    C = sqrt(pi*log(2)/2);
    
    I1 = h(mu_c);
    I2 =  log(2)*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
    
elseif strcmp(modeltype, 'laplace')
    %for a sigmoid link
    C = sqrt(2*log(2));
    I1 = h(mu_c) ;
    I2 = 2*C.*exp(-0.5*mu_y.^2./(sigma2_y+C^2))./sqrt(sigma2_y+C^2);
    
end
I = I1 - I2;

if nargout >1
    dhdp = @(p) -log(p)+log(1-p);
    dI1dx = dhdp(mu_c)*dmuc_dx;
    
    dI2dx =I2.*(0.5*mu_y.^2.*dsigma2y_dx-mu_y.*(sigma2_y+C^2).*dmuy_dx)./((sigma2_y+C^2).^2)-I2./(2*(sigma2_y+C^2)).*dsigma2y_dx;
    
    dIdx = dI1dx - dI2dx;
end

return


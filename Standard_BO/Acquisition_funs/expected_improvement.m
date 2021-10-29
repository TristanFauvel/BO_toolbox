function [EI, dEIdphi] = expected_improvement(muL, varL,dmuL, dvarL, bestf, objective)
sigma_L = sqrt(varL);
if ~isreal(sigma_L)
    print('varL is not positive')
    sigma_L = sqrt(subplus(varL));
end

DL = muL - bestf;


if strcmp(objective, 'min')
    DL = -DL;
end
Z=DL./sigma_L;

normpdfZ =  normpdf(Z);
normcdfZ=normcdf(Z);
EI = (DL.*normcdfZ+ sigma_L.*normpdfZ);%max : Brochu, min : Jones 1998
EI(sigma_L==0)= 0;


if nargout>1
    gaussderZ = -Z.*normpdfZ; %derivative of the gaussian

    dEIdsigmaL = (1-Z.^2).*normpdfZ-Z.*gaussderZ;
    dEIdvarL= dEIdsigmaL./(2*sigma_L);
    dEIdvarL(sigma_L==0)= 0;       
    dEIdmuL =  normcdfZ + Z.*normpdfZ +  gaussderZ ;        
    if strcmp(objective, 'min')
        dEIdmuL=-dEIdmuL;
    end
    dEIdmuL(sigma_L==0)= 0;      

    dEIdphi = ( dEIdmuL*dmuL + dEIdvarL*dvarL)';    
end

return



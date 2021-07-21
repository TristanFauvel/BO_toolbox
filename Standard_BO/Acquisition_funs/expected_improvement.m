function [EI, dEIdphi] = expected_improvement(muL, varL,dmuL, dvarL, bestf, objective)
sigma_L = sqrt(varL);
if ~isreal(sigma_L)
    print('varL is not positive')
    sigma_L = sqrt(subplus(varL));
end

% EI= subplus(DL) + sigma_L.*normpdf(DL./sigma_L) - abs(DL).*normcdf(DL./sigma_L); %Frazier

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

    %     dEIdvarL = (1+DL).*normpdf(Z)-Z.^2.*normpdf(Z)-gaussder(Z)./sigma_L;
    dEIdsigmaL = (1-Z.^2).*normpdfZ-Z.*gaussderZ;
    dEIdvarL= dEIdsigmaL./(2*sigma_L);
    dEIdvarL(sigma_L==0)= 0;       %%%%%%%%%%%%% à vérifier
    
    dEIdmuL =  normcdfZ + Z.*normpdfZ +  gaussderZ ;        
    if strcmp(objective, 'min')
        dEIdmuL=-dEIdmuL;
    end
    dEIdmuL(sigma_L==0)= 0;      

    dEIdphi = ( dEIdmuL*dmuL + dEIdvarL*dvarL)';    
end

% if nargout>1
%     sigma_L = sqrt(varL);    
%     gaussder = @(x) -x*normpdf(x);
%     dEIdvarL =(normpdf(DL./sigma_L).*(1+abs(DL).*DL./varL) - DL./sigma_L.*gaussder(DL./sigma_L))./(2*sigma_L);
%     
%     dEIdmuL =  0.5*(sign(DL)+1) + gaussder(DL./sigma_L) - abs(DL).*normpdf(DL./sigma_L)./sigma_L - sign(DL).*normcdf(DL./sigma_L);        
%     if strcmp(objective, 'min')
%         dEIdmuL=-dEIdmuL;
%     end
%     dEIdphi = ( dEIdmuL*dmuL + dEIdvarL*dvarL);    
% end


return


% subplot(2,1,1)
% errorshaded(1:2500, muL(:), sqrt(varL), 'Color', 'red','DisplayName','Predicted loss');
% subplot(2,1,2)
% plot(EI)
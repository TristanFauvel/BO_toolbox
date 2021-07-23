classdef japan
    properties
        D=2;
        xbounds = [0, 1 ; 0, 1];
        name = 'Japan';
        x
        y
        lengthscale
        kernel
        kernelfun
        domain
        theta
    end
    methods
        function obj = japan(lengthscale, kernelname)
            obj.lengthscale = lengthscale;
            obj.kernel = kernelname;
            load('domain.mat', 'Solin')
            UNPACK_STRUCT(Solin, false)
            
            
            if strcmp(kernelname, 'Matern32')
                if strcmp(lengthscale, 'short')
                    k =1;
                    l =  0.01;
                elseif strcmp(lengthscale, 'long')
                    k =2;
                    l =  0.1;
                end
                theta = [2*log(l) ,0];
            elseif strcmp(kernelname, 'Matern52')
                if strcmp(lengthscale, 'short')
                    k =3 ;
                    l =  0.01;
                elseif strcmp(lengthscale, 'long')
                    k =4;
                    l =  0.1;
                end
                theta = [log(0.5*l^2),0];
            elseif strcmp(kernelname, 'Gaussian')
                if strcmp(lengthscale, 'short')
                    k =5;
                    l =  0.01;
                elseif strcmp(lengthscale, 'long')
                    k =6;
                    l =  0.1;
                end
                theta = [log(0.5*l^2),0.5*obj.D*log(2*pi*l)];
            end
            f = fun{k};
            y = f(:);
            
            [p,q] = meshgrid(domain.x1,domain.x2);
            x = [p(:),q(:)]';
            mask = logical(domain.mask(:));
            % Decrease resolution
            keep = 1:3:numel(y);
            x = x(:,keep);            
            mask = mask(keep);
            y = y(keep);
            %
            obj.x = x(:,mask);
            obj.domain = domain;
            
            
            obj.y = y(mask);
            obj.kernelfun = str2func([kernelname, '_kernelfun']);
            obj.theta =  theta;
        end
        function y = do_eval(obj, xx)
            [a,b] = intersect(obj.x',xx','rows');
            y = obj.y(b);
        end
        function p = plot_function
            graphics_style_paper;
            fig = figure();
            fig.Color = [1 1 1];
            % Color limits
            clims = [-1 1]*max(abs(obj.y(:))); % Adaptive
            hold on
            imagesc(obj.domain.x1,obj.domain.x2,obj.y)
            colormap(cmap)
            plot(obj.domain.poly(1,:),obj.domain.poly(2,:),'-k','LineWidth',1)
            caxis(clims)
            axis ij equal tight off
            drawnow
        end
    end
end

% figure()
% scatter(obj.x(1,:), obj.x(2,:), 5, obj.y)
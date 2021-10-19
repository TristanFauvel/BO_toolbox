% pathname = '/home/tfauvel/Documents/BO_toolbox';
add_bo_module;

load('domain.mat', 'Solin')
UNPACK_STRUCT(Solin, false)
graphics_style_paper;
index= 0;
% For each kernel
fig = figure();
fig.Color = [1 1 1];
for j=1:length(Snames)    
    % For each lengthScale
    for k=1:length(lengthScales)
        index= index+1;
        
        
        f = fun{index};
        % Color limits
        clims = [-1 1]*max(abs(f(:))); % Adaptive
        
        % Crop away the area outside the hexagon (for nicer plots)
    
        
        % Visualize
        subplot(length(lengthScales),length(Snames),(k-1)*length(Snames)+j)
        hold on
        imagesc(domain.x1,domain.x2,f)
        colormap(cmap)
        plot(domain.poly(1,:),domain.poly(2,:),'-k','LineWidth',1)
        caxis(clims)
        text(.5,1.1,Snames{j},'HorizontalAlign','center')
        axis ij equal tight off
        drawnow
        pause(.5)

        
        
    end
end
%RGB = ind2rgb(ceil(64*(f-min(clims))/(max(clims)-min(clims))),parula(64)); %parula(64)

%         RGB = reshape(RGB,[],3);
%         ind = ~domain.mask(:);
%         RGB([ind ind ind]) = ones(sum(ind),3);
%         RGB = reshape(RGB,size(f,1),size(f,2),3);


nobj = 6;
objectives = cell(1,nobj);
for i = 1:nobj
    objectives{i} = 'japan';
end
kernelnames = {'Matern32', 'Matern32','Matern52','Matern52','Gaussian','Gaussian'};
lengthscales = {'long', 'short','long', 'short','long', 'short'};
meanfun = @constant_mean;
nmean_hyp = 1;
options_theta.verbose = 1;
options_theta.method= 'lbfgs';
for j = 2:nobj
    bias = 0;
    objective = [objectives{j}, '_', kernelnames{j}, '_',lengthscales{j}];
    [x, y, theta.cov, lb, ub, hyp_lb, hyp_ub, kernelfun] = load_benchmarks_active_learning_grid(objectives{j}, kernelnames{j}, lengthscales{j});
    update = 'cov';
    theta.mean = 0;
    theta.cov = theta.cov';
    init_guess = [theta.cov;theta.mean];
    ncov_hyp = numel(theta.cov);
    hyp = multistart_minConf(@(hyp)minimize_negloglike(hyp, x, y, kernelfun, meanfun, ncov_hyp, nmean_hyp, update), [hyp_lb';0], [hyp_ub';0],10, init_guess, options_theta);
    theta.cov = hyp(1:ncov_hyp);
    theta.mean = hyp(ncov_hyp+1:ncov_hyp+nmean_hyp);
    
    disp([objective, ' : ', num2str(theta.cov')])
    
end

% japan_Matern32_long : -2.1738    -0.88348
% japan_Matern32_short : -3.4928   -0.099319
% japan_Matern52_long : -2.7098     -2.1358
% japan_Matern52_short : -3.5252     0.55026
% japan_Gaussian_long : 7.1851    -0.31328
% japan_Gaussian_short : 8.7792    -0.58559



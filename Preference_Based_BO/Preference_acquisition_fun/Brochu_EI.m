function [x_duel1, x_duel2, new_duel] = Brochu_EI(theta, xtrain_norm, ctrain, model, post, approximation)
% Expected Improvement, as proposed by Brochu (2010)

D = size(xtrain_norm,1)/2;
n = size(xtrain_norm,2);
%% Find the maximum of the value function
options.method = 'lbfgs';

ncandidates= 5;

condition = model.condition;

x = [xtrain_norm(1:D,:), xtrain_norm((D+1):end,:)];
[g_mu_c,  g_mu_y] = model.prediction(theta, xtrain_norm, ctrain, [x;condition.x0*ones(1,2*n)], post);
[max_mu_y,b]= max(g_mu_y);
x_duel1 = x(:,b);


% x_duel2 = minFuncBC(@(x)expected_improvement_for_classification(theta, xtrain_norm, x, ctrain, lb_norm, ub_norm, kernelfun,kernelname, x0, modeltype), x_init, lb_norm, ub_norm, options);
% x_duel2 = multistart_minfuncBC(@(x)expected_improvement_for_classification(theta, xtrain_norm, x, ctrain, lb_norm, ub_norm, kernelfun,kernelname, x0, x_best, modeltype), lb_norm, ub_norm, ncandidates, options);

init_guess = x_duel1;
x_duel2 = multistart_minConf(@(x)expected_improvement_preference(theta, xtrain_norm, x, ctrain, max_mu_y, model, post), model.lb_norm, model.ub_norm, ncandidates, init_guess, options);

x_duel1 = x_duel1.*(model.max_x(1:D)-model.min_x(1:D)) + model.min_x(1:D);
x_duel2 = x_duel2.*(model.max_x(D+1:2*D)-model.min_x(D+1:2*D)) + model.min_x(D+1:2*D);

new_duel = [x_duel1;x_duel2];

% if isequal(x_duel1, x_duel2)
%     error('x_duel1 is the same as x_duel2')
%
%     x = linspace(0,1,100);
%     [mu_c,  mu_g, sigma2_g] = model.prediction(theta, xtrain_norm, ctrain, [x;condition.x0*ones(1,100)], post);
%
%     [~,  ymax1] = model.prediction(theta, xtrain_norm, ctrain, [x_duel1;condition.x0], post);
%
%     x_duel2 = multistart_minConf(@(x)compute_bivariate_expected_improvement(theta, xtrain_norm, x, ctrain, model, x_duel1, post), model.lb_norm, model.ub_norm, 15,init_guess, options);
%         [~,  ymax2] = model.prediction(theta, xtrain_norm, ctrain, [x_duel2;condition.x0], post);
%
%      xx = sort(xtrain_norm,1);
%      xx1 = xx(1,:);
%      xx2 = xx(2,:);
%     graphics_style_paper;
%     mr = 2;
%     mc = 3;
%     fig=figure('units','centimeters','outerposition',1+[0 0 16 0.6*fheight(mr)]);
%     fig.Color =  [1 1 1];
%     i = 0;
%     tiledlayout(mr,mc, 'TileSpacing' , 'tight', 'Padding', 'tight')
%
%     nexttile(1, [2,1])
%     i=1;
%     h1 = plot(1:size(xtrain_norm,2), xx1,'Color',  C(1,:),'linewidth',linewidth); hold on;
%     h2 = plot(1:size(xtrain_norm,2), xx2,'Color',  C(2,:),'linewidth',linewidth); hold off;
%     legend([h1, h2], '$x_1$', '$x_2$')
%     xlabel('Iteration', 'Fontsize', Fontsize)
%     ylabel('$x$', 'Fontsize', Fontsize)
%     legend box off
%     box off
%     xlim([1,size(xtrain_norm,2)])
%     ytick = get(gca,'YTick');
%     set(gca,'YTick', linspace(min(ytick), max(ytick), 3),  'Fontsize', Fontsize)
%     text(-0.08,1,['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
%     figure_path = '/home/tfauvel/Documents/PhD/Figures/Thesis_figures/Chapter_1/';
%
%
%     nexttile(2, [1,2])
%     i=i+1
%     %     plot(x, g_mu_c, C(1,:), linewidth); hold on;
%     p = plot_gp(x, mu_g, sigma2_g, C(1,:), linewidth); hold on;
%     vline(x_duel1,'Linewidth',linewidth, 'ymax', ymax1,  'LineStyle', '--', ...
%         'Linewidth', 1, 'Color', C(1,:)); hold on;
%     vline(x_duel2,'Linewidth',linewidth, 'ymax', ymax2,  'LineStyle', '--', ...
%         'Linewidth', 1,'Color', C(2,:)); hold off;
%
%     %     plot(x,g, 'Color',  C(1,:),'linewidth',linewidth); hold on
%     xlabel('$x$', 'Fontsize', Fontsize)
%     ylabel('$g(x)$', 'Fontsize', Fontsize)
%     set(gca,'XTick',[0 0.5 1], 'Fontsize', Fontsize)
%     ytick = get(gca,'YTick');
%     set(gca,'YTick', linspace(min(ytick), max(ytick), 3))
%     box off
%     legend(p, 'Posterior GP')
%     legend box off
%     %title('Inferred value function $g(x)$','Fontsize', Fontsize)
%     text(-0.08,1,['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
%      [xt,b] = sort([0,x_duel1, x_duel2, 1]);
%     xticks(xt)
%     lgs = {'0', '$x_1$', '$x_2$','1'};
%     xticklabels(lgs(b))
%
%     nexttile(5, [1,2])
%     L1 = expected_improvement_preference(theta, xtrain_norm, x, ctrain, max_mu_y, model, post)
%     L1 = -L1;
%     h1 = plot(x, L1, 'Color',  C(1,:),'linewidth',linewidth); hold on;
%
%     set(gca, 'Fontsize', Fontsize)
%     box off
%         vline(x_duel1,'Linewidth',linewidth, 'ymax', max(L1),  'LineStyle', '--', ...
%         'Linewidth', 1, 'Color', C(1,:)); hold on;
%     L2 = compute_bivariate_expected_improvement(theta, xtrain_norm, x, ctrain, model, x_duel1, post)
%     L2 = -L2;
%     h2 = plot(x, L2, 'Color',  C(2,:),'linewidth',linewidth);
%         xlabel('$x$', 'Fontsize', Fontsize)
%     ylabel('$\alpha(x)$', 'Fontsize', Fontsize)
%      box off
%         vline(x_duel2,'Linewidth',linewidth, 'ymax', max(L2),  'LineStyle', '--', ...
%         'Linewidth', 1,'Color', C(2,:)); hold off;
%
%     legend([h1, h2], 'EI', 'Bivariate EI')
%     legend box off
%     i=3
%     text(-0.08,1.15,['$\bf{', letters(i), '}$'],'Units','normalized','Fontsize', letter_font)
%     [xt,b] = sort([0,x_duel1, x_duel2, 1]);
%     xticks(xt)
%     lgs = {'0', '$x_1$', '$x_2$','1'};
%     xticklabels(lgs(b))
%     set(gca, 'ylim', [0, max([L1(:);L2(:)]')]);
%          ytick = get(gca,'YTick');
%     set(gca,'YTick', linspace(min(ytick), max(ytick), 3))
%
%
%     figname  = 'Brochu_EI_pathology';
%     folder = [figure_path,figname];
%     savefig(fig, [folder,'/', figname, '.fig'])
%     exportgraphics(fig, [folder,'/' , figname, '.pdf']);
%     exportgraphics(fig, [folder,'/' , figname, '.png'], 'Resolution', 300);
%
%
% end

end

function [EI, dEI_dx] = expected_improvement_preference(theta, xtrain_norm, x, ctrain, max_mu_y, model, post)

[D,n]= size(x);
[g_mu_c,  g_mu_y, g_sigma2_y, g_Sigma2_y, dmuc_dx, dmuy_dx, dsigma2_y_dx] = model.prediction(theta, xtrain_norm, ctrain, [x;model.condition.x0*ones(1,n)], post);

dmuc_dx = dmuc_dx(1:D,:);
dmuy_dx = dmuy_dx(1:D,:);
dsigma2_y_dx = dsigma2_y_dx(1:D,:);

g_sigma_y = sqrt(g_sigma2_y);
%% Find the maximum of the value function

sigma_y = sqrt(g_sigma2_y);
d = (g_mu_y - max_mu_y)./sigma_y;
d(sigma_y==0)=0;

normpdf_d =  normpdf(d);
normcdf_d= normcdf(d);

EI = (g_mu_y - max_mu_y).*normcdf_d+ sigma_y.*normpdf_d;%Brochu

EI(sigma_y==0)= 0;

if nargout>1
    gaussder_d = -d.*normpdf_d; %derivative of the gaussian
    dsigma_y_dx = dsigma2_y_dx./(2*g_sigma_y);
    dsigma_y_dx(g_sigma2_y==0,:) = 0;
    dd_dx = (-dmuy_dx.*g_sigma_y - (max_mu_y - g_mu_y).*dsigma_y_dx)./g_sigma2_y;
    dd_dx(g_sigma2_y==0,:) = 0;
    dEI_dx = dmuy_dx.*normcdf_d - (max_mu_y - g_mu_y).*normpdf_d.*dd_dx + dsigma_y_dx.*normpdf_d +g_sigma_y.*gaussder_d.*dd_dx;
    dEI_dx = -squeeze(dEI_dx);%This is because the goal is to maximize EI, and I use a minimization algorithm
end

EI = -EI; %This is because the goal is to maximize EI, and I use a minimization algorithm
end

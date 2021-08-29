function fig =  plot_benchmarks_histograms(final_values, AUCs, legends, b)

nacq = size(final_values, 1);
nobj = size(final_values, 2);
nreps = size(final_values, 3);

for j =1:nobj
    v = squeeze(final_values(:,j,:));
    final_values(:,j,:) = reshape(zscore(v(:)), nacq, nreps);
    
    v = squeeze(AUCs(:,j,:));
    AUCs(:,j,:) = reshape(zscore(v(:)), nacq, nreps);
    
    %     final_values(:,j,:) = zscore(final_values(:,j,:));
    %     AUCs(:,j,:) = zscore(AUCs(:,j,:));
end
 
graphics_style_paper;
mr = 2;
mc = 1;
legend_pos = [-0.18,1];

% fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
% fig.Color =  [1 1 1];
% layout = tiledlayout(mr,mc, 'TileSpacing', 'compact', 'padding','compact');
% i = 0;
% 
% short_acq_names= T(any(T.acq_funs == acq_funs,2),:).short_names; 
% 
% nexttile();
% for i=1:nacq
%     vals = squeeze(final_values(b(i),:,:));
%    scatter(i*ones(1,nobj*nreps), vals(:), 15, 'filled') ; hold on;
% end
% set(gca, 'Fontsize', Fontsize, 'XtickLabel', short_acq_names(b))
% 
% nexttile();
% for i=1:nacq
%     vals = squeeze(AUCs(b(i),:,:));
%    scatter(i*ones(1,nobj*nreps), vals(:), 15, 'filled') ; hold on;
% end
% set(gca, 'Fontsize', Fontsize, 'XtickLabel', legends(b))

%%

fig=figure('units','centimeters','outerposition',1+[0 0 fwidth fheight(mr)]);
fig.Color =  [1 1 1];
layout = tiledlayout(mr,mc, 'TileSpacing', 'compact', 'padding','compact');
i = 0;

 
nexttile();
for i=1:nacq
    vals = squeeze(final_values(b(i),:,:));
    [f,xi] = ksdensity(vals(:));
    plot(xi,f,'LineWidth', linewidth); hold on;
end
set(gca, 'xlim', [-4,4])
box off
legend boxoff
legend(legends(b));

nexttile();
for i=1:nacq
    vals = squeeze(AUCs(b(i),:,:));
    [f,xi] = ksdensity(vals(:));
    plot(xi,f,'LineWidth', linewidth); hold on;
end
set(gca, 'xlim', [-4,4])

legend(legends(b));
box off
legend boxoff

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



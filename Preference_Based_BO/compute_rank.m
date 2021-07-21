function [ranks, average_ranks, rmatrix]= compute_rank(scores, ninit)
nacq = numel(scores);
[nrepets, maxiter] =  size(scores{1});
rank_matrix = NaN(nrepets, maxiter,nacq);
rmatrix = NaN(nacq, nrepets*(maxiter-ninit));
M = cell2mat(arrayfun(@(x)permute(x{:},[1 3 2]),scores,'UniformOutput',false));
for i = 1:nrepets
    for j = 1:maxiter
        [~, ordering]= sort(M(i,:,j));
        rank_matrix(i,j,:) = ordering;
    end
end
step = 1;
for i =1:nacq
    ranks{i}= rank_matrix(:,ninit+1:step:end,i);
    average_ranks{i} = mean(ranks{i});
    
    t = ranks{i};
    rmatrix(i,:) = t(:);
end

%%

% rank_matrix = NaN(maxiter,nacq);
%
% for j = 1:maxiter
%     [~, ordering]= sort(mean(M(:,:,j),1));
%     rank_matrix(j,:) = ordering;
% end
%
end
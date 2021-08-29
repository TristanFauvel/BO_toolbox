function [t, Best_ranking, AUC_ranking,b, signobj, ranking, final_values, AUCs] = ranking_analysis(data_path, names, objectives, algos, nreps, maxiter, suffix)

nobj = numel(objectives);
nacq = numel(algos);
graphics_style_paper;

rng(1);
benchmarks_results = cell(1,nobj);
scores = cell(1,nacq);
final_values = zeros(nacq,nobj, nreps);
AUCs = zeros(nacq,nobj, nreps);

for j = 1:nobj
    %     objective = objectives{j};
    objective = char(objectives(j));
    
    for a = 1:nacq
        acquisition = algos{a};
        filename = [data_path,'/',objective, '_',acquisition,suffix];
%         try
            load(filename, 'experiment');
            score= cell2mat(experiment.(['score_', acquisition])');
             scores{a} = score;
            final_values(a,j,:) = score(:,end); 
            AUCs(a,j,:) = mean(score,2); 
%         catch
%             scores{a} = NaN(nreps, maxiter);
%         end
        
    end
    benchmarks_results{j} = scores;
    %     [ranks, average_ranks]= compute_rank(scores, ninit);
end

alpha = 5e-4;
%% Partial ranking based on Mann-Withney t-test at alpha = 5e-4 significance
R_best = NaN(nobj, nacq, nacq);
R_AUC = NaN(nobj, nacq, nacq);
for k = 1:nobj
    S = benchmarks_results{k};
    for i = 1:nacq
        best_i = S{i};
        best_i = best_i(:,end);
        score_i = S{i};
        AUC_i= mean(score_i,2);
        for j = 1:nacq
            try
                best_j = S{j};
                best_j = best_j(:,end);
                [p,h] = ranksum(best_i, best_j, 'tail','right', 'alpha', alpha); % Test whether i beats j
                R_best(k, i, j) = h;
                
                score_j = S{j};
                AUC_j = mean(score_j, 2);
                
                [p,h] = ranksum(AUC_i, AUC_j, 'tail','right', 'alpha', alpha); % Test whether i beats j
                R_AUC(k, i, j) = h;
            catch
                disp(k)
                disp(i)
            end
        end
    end
end
partial_ranking = squeeze(sum(R_best,3));


borda_scores = NaN(size(partial_ranking));
for i=1:nobj
    partial_ranking(i,:)  = get_ranking(partial_ranking(i,:));
    borda_scores(i,:)  = get_Borda_from_ranking(partial_ranking(i,:));
end

%% Ties from the previous step are broken based on the Area Under Curve
for k = 1:nobj
    ranks_k = partial_ranking(k,:);
    nranks = sort(unique(ranks_k));
    for i = 1:numel(nranks)
        r = nranks(i);
        eq = (ranks_k == r); % functions having the same rank
        if sum(eq) >1 % several functions have the same rank
            %auc_comparisons = AUC_ranking(k,eq); %% Use all the AUC
            %comparisons
            auc_comparisons = squeeze(sum(R_AUC(k,eq,eq),3)); % Only use the comparisons within the ties %%
            borda_modifier = get_Borda_from_ranking(get_ranking(auc_comparisons));
            borda_scores(k,eq) = borda_scores(k,eq)+borda_modifier;           
        end
    end
end

Borda_score = sum(borda_scores,1);
ranking = get_ranking(Borda_score);

[a,b] = sort(ranking);
ordered_names = names(b,:);
t = table(ordered_names, ranking(b)', Borda_score(b)', 'VariableNames', {'Acquisition rule', 'Rank', 'Borda score'});

Best_ranking = squeeze(sum(R_best,1))/nobj;
Best_ranking = Best_ranking(b,b);
AUC_ranking = squeeze(sum(R_AUC,1))/nobj;
AUC_ranking =  AUC_ranking(b,b);


% objectives with the most significant difference between algos
[s,a] = sort(sum(borda_scores, 2));
 signobj = flipud(a(s>0));
 
  signobj = sum(borda_scores,2)>0;

 
% objectives that agree the most with the final ranking
% [s2,a] = sort(mean((borda_scores- (nacq-ranking)).^2,2));
% 
%  signobj = a;

end

function U  = rank_with_ties(V)
U = NaN(size(V));
d = unique(sort(V));
for i =1:numel(d)
    U(V == d(i)) =i-1;
end
end

function U  = get_ranking(Borda_score)
V = Borda_score;
U = NaN(size(V));
d = fliplr(unique(sort(V)));
s= 1;
for i =1:numel(d)
    U(V == d(i)) = s;
    s = s + sum(V == d(i));
end
end

function B  = get_Borda_from_ranking(ranking)
V = ranking;
B= NaN(size(V));
ranks = sort(unique(V),'descend');
s = 0;
for r = ranks
    B(V == r) = s;
    s = s+sum(V==r);
end
end


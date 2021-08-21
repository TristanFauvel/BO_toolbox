function t = ranking_analysis_w_vs_wo_condition(data_dir_w, data_dir_wo, names, objectives, algos, nreps, maxiter)

nobj = numel(objectives);
nacq = numel(algos);
graphics_style_paper;

rng(1);
benchmarks_results_wo = cell(1,nobj);
scores_wo = cell(1,nacq);
for j = 1:nobj
    %     objective = objectives{j};
    objective = char(objectives(j));
    
    for a = 1:nacq
        acquisition = algos{a};
        filename = [data_dir_wo,'/',objective, '_',acquisition];
        %         try
        load(filename, 'experiment');
        UNPACK_STRUCT(experiment, false)
        scores_wo{a} = cell2mat(eval(['score_', acquisition])');
        
        %         catch
        %             scores_wo{a} = NaN(nreps, maxiter);
        %         end
        
    end
    benchmarks_results_wo{j} = scores_wo;
    %     [ranks, average_ranks]= compute_rank(scores, ninit);
end

benchmarks_results_w = cell(1,nobj);
scores_w = cell(1,nacq);
for j = 1:nobj
    objective = char(objectives(j));  
    
    for a = 1:nacq
        acquisition = algos{a};
        filename = [data_dir_w,'/',objective, '_',acquisition];
        load(filename, 'experiment');
        UNPACK_STRUCT(experiment, false)
        scores_w{a} = cell2mat(eval(['score_', acquisition])');     
    end
    benchmarks_results_w{j} = scores_w;
end

nacq = 2;
alpha = 5e-4;
%% Partial ranking based on Mann-Withney t-test at alpha = 5e-4 significance
R_best = NaN(nobj, nacq, nacq);
R_AUC = NaN(nobj, nacq, nacq);
for k = 1:nobj
    S_w = benchmarks_results_w{k};
    S_wo = benchmarks_results_wo{k};
    S{1} = S_w{1};
    S{2} = S_wo{1};
    
    for i = 1:2
        best_i = S{i};
        best_i = best_i(:,end);
        score_i = S{i};
        AUC_i= mean(score_i,2);
        for j = 1:2
            best_j = S{j};
            best_j = best_j(:,end);
            [p,h] = ranksum(best_i, best_j, 'tail','right', 'alpha', alpha); % Test whether i beats j
            R_best(k, i, j) = h;
            
            score_j = S{j};
            AUC_j = mean(score_j, 2);
            
            [p,h] = ranksum(AUC_i, AUC_j, 'tail','right', 'alpha', alpha); % Test whether i beats j
            R_AUC(k, i, j) = h;
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

[~,b] = sort(ranking);
names = {'With conditioning', 'Without conditioning'};
ordered_names = names(b);
t = table(ordered_names(:), ranking(b)', Borda_score(b)', 'VariableNames', {'Acquisition rule', 'Rank', 'Borda score'});
% table2latex(t, 'PBO_benchmark_results')
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

% graphics_style_paper;
% 
% k=1;
% figure()
% p1 = plot_gp(1:maxiter, mean(cell2mat(benchmarks_results_w{k}))', var(cell2mat(benchmarks_results_w{k}))', C(1,:), linewidth); hold on;
% p2 = plot_gp(1:maxiter, mean(cell2mat(benchmarks_results_wo{k}))',var(cell2mat(benchmarks_results_wo{k}))', C(2,:), linewidth); hold on;
% legend([p1,p2], {'With conditioning', 'Without conditioning'})
% 

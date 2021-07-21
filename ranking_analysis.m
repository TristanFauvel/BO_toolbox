function t = ranking_analysis(data_path, names, objectives, algos, nreps, maxiter)

nobj = numel(objectives);
nacq = numel(algos);
graphics_style_paper;

rng(1);
ninit = 5;
benchmarks_results = cell(1,nobj);
scores = cell(1,nacq);
for j = 1:nobj
%     objective = objectives{j};
    objective = char(objectives(j));

    if strcmp(objective, 'goldpr')
        options.semilogy = true; %true;
    else
        options.semilogy = false;
    end
    for a = 1:nacq
        acquisition = algos{a};
        filename = [data_path,'/',objective, '_',acquisition];
        try
            load(filename, 'experiment');
            UNPACK_STRUCT(experiment, false)
            legends{a}=[names{a}];
            n=['a',num2str(a)];
            
            scores{a} = cell2mat(eval(['score_', acquisition])');
            
        catch
            scores{a} = NaN(nreps, maxiter);
        end
        
    end
    benchmarks_results{j} = scores;
%     [ranks, average_ranks]= compute_rank(scores, ninit);
end


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
            [p,h] = signrank(best_i, best_j, 'tail','right', 'alpha', 5e-4); % Test whether i beats j
            R_best(k, i, j) = h;
            
            score_j = S{j};
            AUC_j = mean(score_j, 2);
            
            [p,h] = signrank(AUC_i, AUC_j, 'tail','right', 'alpha', 5e-4); % Test whether i beats j
            R_AUC(k, i, j) = h;
            catch 
                disp(k)
                disp(i)
            end
        end
    end
end

squeeze(R_best(1, :, :))
squeeze(R_AUC(1, :, :))
squeeze(R_best(2, :, :))
squeeze(R_AUC(2, :, :))

partial_ranking = squeeze(sum(R_best,3));

AUC_ranking = squeeze(sum(R_AUC,3));
total_ranking = partial_ranking;

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
            
            total_ranking(k,eq) = total_ranking(k,eq) + rank_with_ties(auc_comparisons);
            
        end
    end
end

Borda_score = sum(total_ranking,1);
ranking = get_ranking(Borda_score);

[a,b] = sort(ranking);
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


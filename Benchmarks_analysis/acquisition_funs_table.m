acq_funs = categorical({ 'MVT', 'maxvar_ALD','KSS_ALD','UCB_binary','BKG', 'maxvar_challenge', 'EIIG', 'Dueling_UCB', 'MaxEntChallenge','DTS','random_acquisition_pref', ...
    'kernelselfsparring', 'bivariate_EI', 'Brochu_EI', 'Thompson_challenge', 'kernelselfsparring_tour','random_acquisition_tour', ...
      'cBKG', 'UCB_ALD', 'TS_ALD','UCB_rands','TSx_rands', 'randx_ALDs', 'rand_acq', ...
    'random', 'pref_randx_ALDs','KSSx_rands','TME_sampling_binary', 'BALD_grid', ...
    'Variance_gradient_grid', 'TS_maxvar', 'TS_binary', 'random_acquisition_binary', ...
      'UCB_binary_latent', 'EI_Tesch', 'bivariate_EI_binary', 'active_sampling_binary',  'cPKG', 'UCB_binary_latent_mean'}');

names = categorical({'Maximum Variance Tournament', 'MUC-ALD', 'KSS-ALD','UCB$_\Phi$','Binary Knowledge Gradient','Maximally Uncertain Challenge', 'EIIG (Benavoli et al 2020)', 'Dueling UCB (Benavoli et al 2020)',...
    'MaxEntChallenge','Duel TS (Gonzalez et al 2017)', ...
    'Random', 'Kernel Self-Sparring (Sui et al 2017)',  ...
    'Bivariate EI (Nielsen et al 2015)', 'Expected Improvement (Brochu et al 2010)', 'Dueling Thompson (Benavoli et al 2020)', ...
    'Kernel Self-Sparring', 'Random', 'Binary Knowledge Gradient',  'Upper Credible Bound with Active Learning by Disagreement', ...
    'Thompson Sampling with Active Learning by Disagreement','UCB (random context)','Thompson Sampling (random context)', 'BALD (Houlsby et al 2011)', 'Random', ...
    'Random',  'BALD (Houlsby et al 2011)','Kernel Self-Sparring', 'Total Marginal Entropy sampling', 'Bayesian Active Learning by Disagreement',...
    'Epistemic Uncertainty gradient','Thompson Sampling with Maximum Uncertainty','Thompson sampling', 'Random', ...
    'UCB$_f$', 'Binary Expected Improvement', 'Bivariate Expected Improvement', 'Bayesian Active Learning by Disagreement', ...
    'cPKG', 'UCB_binary_latent_mean'}');

names_citations = categorical({'Maximum Variance Tournament','MUC-ALD','KSS-ALD','UCB$_\Phi$', 'Binary Knowledge Gradient', 'Maximally Uncertain Challenge', 'EIIG \citep{Benavoli2020}', 'Dueling UCB \citep{Benavoli2020}', 'MaxEntChallenge', ...
    'Duel Thompson Sampling \citep{Gonzalez2017}','KernelSelfSparring \citep{Sui2017}','Random', ...
    'Bivariate Expected Improvement \citep{Nielsen2015}', 'Expected Improvement \citep{Brochu2010a}',...
    'Dueling Thompson \citep{Benavoli2020}', 'Random', 'KernelSelfSparring \citep{Sui2017}', ...
    'cBKG', 'UCB-ALD','TS-ALD','UCB (random context)','Thompson Sampling (random context)', 'BALD \citep{Houlsby2011}', 'Random','Random','pref_randx_ALDs','KSSx_rands', 'TME sampling', ...
    'BALD \citep{Houlsby2011}', 'Epistemic Uncertainty gradient','TS-MU','Thompson sampling', 'Random', ...
    'UCB$_f$ \citep{Tesch2013}', 'Binary Expected Improvement \citep{Tesch2013}', 'Bivariate Expected Improvement', ...
    'BALD', 'cPKG', 'UCB_binary_latent_mean'}');

short_names=  categorical({'MUC', 'MUC-ALD','KSS-ALD','UCB$_\Phi$', 'BKG', 'MUC', 'EIIG', 'UCB', 'MEC','DTS','Random', 'KSS', 'BEI', 'EI', 'Dueling TS', 'KSS','Random', ...
    'cBKG', 'UCB-ALD','TS-ALD','UCB','TS', 'BALD', 'Random','Random','BALD','KSS','TME sampling', 'BALD', ...
    'Epistemic Uncertainty gradient','TS-MU','TS', 'Random', ...
    'UCB$_f$', 'Binary EI', 'Bivariate EI','BALD',  'cPKG', 'UCB_binary_latent_mean'}');

T = table(acq_funs, names,names_citations, short_names);
save('/home/tfauvel/Documents/BO_toolbox/Acquisition_funs_table','T')



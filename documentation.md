# BO_toolbox documentation
>Copyright Tristan Fauvel (2021)
>This software is distributed under the MIT License. Please refer to the file LICENCE.txt included for details

**BO_toolbox** is a **Matlab** toolbox for Bayesian optimization, with a focus on binary and preferential Bayesian optimization.
This is based on **GP_toolbox** (https://github.com/TristanFauvel/GP_toolbox).

## Installation
  Simply add **BO_toolbox** to your Matlab path

## Organization

The code is divided into 5 main parts:
  - Active learning
  - Standard BO
  - Binary BO
  - Preferential BO
  - Benchmarks analysis: some code to analyze synthetic experiments on benchmarks.

## List of variables

- `objective`: objective function  
- `task`: whether the goal is to maximize the 'max'
- `identification`: identification step
- `maxiter`: number of iteration
- `nopt`: number of time steps before starting using the specificied acquisition function.
- `ninit`:
- `update_period`
- `hyps_update`: wether all `all` hyperparameters should be updated, only the kernel hyperparameters (`cov`), the mean (`mean`) or `none`.
- `acquisition_fun`: acquisition function.
- `ns`
- `noise`: amount of noise in function evaluation (for standard BO)


## User guide

For each type of optimization problem, you can create an optimization object using either `standard_BO`, `binary_BO`, `preferential_BO`, `active_learning`. The surrogate model is a Gaussian process.

## Optimization

### Standard BO
####  List of acquisition functions :

### Binary BO:
####  List of acquisition functions :
  - `BKG()` :  Binary knowledge gradient
  - `EI_Tesch()` : Binary Expected improvement, as defined by Tesch et al (2013)
  - `TS_binary` : Thompson sampling
  - `random_acquisition` : selects inputs uniformly at random

### Preferential BO:
####  List of acquisition functions :
- `bivariate_EI` : Bivariate Expected Improvement (Nielsen et al, 2015).
- `Brochu_EI` : Expected Improvement (Brochu et al, 2010))
- `Dueling UCB` : Dueling Upper Credible Bound, as defined by (Benavoli et al, 2020)
-  `kernelselfsparring`: KernelSelfSparring (Sui et al, 2017)
- `EIIG`: EIIG (Benavoli et al, 2020)
- `DTS`: Duel Thompson Sampling (Gonzalez et al, 2017)
- `Thomspon_challenge` :  Dueling Thompson (Benavoli et al 2020).
- `PKG`: preferential knowledge gradient

### Active learning:
Active learning for GP classification and preference learning models.
####  List of acquisition functions :
- `BALD` : Bayesian Active Learning by Disagreement (Houlsby et al, 2011)

## Reference
If you use this software, please reference it as follows : Tristan Fauvel (2021) BO_toolbox, a Matlab toolbox for Bayesian optimization.

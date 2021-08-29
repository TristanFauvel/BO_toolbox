# BO_toolbox
 A Matlab toolbox for Bayesian optimization.

## Reference

If you use this software, please reference it as follows : Tristan Fauvel (2021) BO_toolbox, a Matlab toolbox for Bayesian optimization.

## Features
This toolbox was primarily intended to perform preferential Bayesian optimization. The toolbox features:
* Bayesian optimization with continuous outputs
* Bayesian optimization with binary outputs
* Preferential Bayesian optimization
* State-of-the art acquisition functions :
  * For standard Bayesian optimization :
    * Thompson sampling
    * Expected improvement
    * GP-UCB
  * For optimization with binary outputs
  * For preference-based optimization
    * Bivariate Expected Improvement
    * Expected improvement
    * KernelSelfSparring
* This toolbox is based on the GP_toolbox (https://github.com/TristanFauvel/GP_toolbox), which uses state-of-the art methods for approximate sampling from GP posteriors and for preference learning with GP.

## Installation
* Simply add BO_toolbox to your Matlab path

## User guide
* Explanations are provided in the /Examples subfolders

## License
This software is distributed under the MIT License. Please refer to the file LICENCE.txt included for details.

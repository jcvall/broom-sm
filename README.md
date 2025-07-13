<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/broom-sm.svg?branch=main)](https://cirrus-ci.com/github/<USER>/broom-sm)
[![ReadTheDocs](https://readthedocs.org/projects/broom-sm/badge/?version=latest)](https://broom-sm.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/broom-sm/main.svg)](https://coveralls.io/r/<USER>/broom-sm)
[![PyPI-Server](https://img.shields.io/pypi/v/broom-sm.svg)](https://pypi.org/project/broom-sm/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/broom-sm.svg)](https://anaconda.org/conda-forge/broom-sm)
[![Monthly Downloads](https://pepy.tech/badge/broom-sm/month)](https://pepy.tech/project/broom-sm)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/broom-sm)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# broom-sm

Python implementation of the R broom package (and other helpful functions) with a wrapper around Statsmodels.

The broom-sm module was inspired from David Robinson's broom package (r-broom) that tidy's statistical results. I built this package to be used with and compliment the pyjanitor package (python-pyjanitor) created by Eric Ma.

The extra-sm module was inspired from David Robinson's book "Introduction to Empirical Bayes".

The goal of this project is to provide tidy summaries for the following statistical models in Statsmodel & Scipy, with some emperical bayesian functions for analysis:

**broom-sm** 

+ OLS Regression
+ GLM-Logit Regression
+ GLM-Poisson Regression
+ GLM-Gamma Regression
+ GLM-Quantile Regression
+ GLM-Mixed Linear Model Regression
+ Elastic Net Regression
+ Survival Regression
+ ARIMA Time Series
+ ANOVA
+ Kruskal-Wallis ANOVA
+ Chi Square
+ Pearson & Spearman Correlation
+ Bootstrap
+ Permutations
+ Contingentcy Tables
+ Goodness of fit (distributions)

**extra-sm (emperical bayes functions)**

* eb_fit_prior()
* add_eb_estimate()
* add_eb_prop_test()
* eb_fit_mixmodel()
* py_beta()
* py_binom()
* py_gamma()
* py_possion()
* py_normal()
* eb_simulation()

### Install Package

```
pip install git+https://github.com/jcvall/broom-sm.git
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.

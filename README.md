## Description
This repository contains code related to “[Factor-augmented tree ensembles](https://arxiv.org/abs/2111.14000)”.

This article proposes an extension for standard time-series regression tree modelling to handle predictors that show irregularities such as missing observations, periodic patterns in the form of seasonality and cycles, and non-stationary trends. In doing so, this approach permits also to enrich the information set used in tree-based autoregressions via unobserved components. Furthermore, this manuscript also illustrates a relevant approach to control over-fitting based on ensemble learning and recent developments in the jackknife literature. This is strongly beneficial when the number of observed time periods is small and advantageous compared to benchmark resampling methods. Empirical results show the benefits of predicting equity squared returns as a function of their own past and a set of macroeconomic data via factor-augmented tree ensembles, with respect to simpler benchmarks. As a by-product, this approach allows to study the real-time importance of economic news on equity volatility.

## Citation
If you use this code or build upon it, please use the following (bibtex) citation:
```bibtex
@misc{pellegrino2022factoraugmented,
      title={Factor-augmented tree ensembles}, 
      author={Filippo Pellegrino},
      year={2022},
      eprint={2111.14000},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

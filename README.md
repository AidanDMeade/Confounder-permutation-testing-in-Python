# Confounder-permutation-testing-in-Python

## Overview

A modal agnostic confounder-restricted permutation testing approach for investigating the influence of categorical confounders on model performance. This is used as part of the paper "Detecting the influence of clinical confounder bias within machine learning models - a study of ANCA-associated vasculitis autoantibody subtype with serum-FTIR spectroscopy" published in Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy.

A PCA-SVM binary classification model is used as an illustrative example. 
The framework can be readily adapted to other machine learning models.
This project consists of two scripts that are executed in sequence: 
1. Run **permutation_testing_code.py** 
2. Run **permutation_testing_ks_analysis.py** 


# Permutation testing

The script **permutation_testing_code.py** implements a high-throughput PCA-SVM classification workflow with:

- StratifiedGroupKFold cross-validation
- PCA component sweep (1 → ncomp)
- Hyperparameter optimisation via GridSearchCV
- Full metric logging (fold-wise + per-sample)
- A restricted permutation test designed using methods adapted from work by Neto et al. (Chaibub Neto et al., 2019a, 2019b). 

The workflow performs:

- Null distribution visualisation for 5 categorical confounders
- Observed AUC comparison against null
- Kolmogorov-Smirnov (KS) tests 
- ECDF visualisation
- Automated multi-page PDF generation
- Publication-ready figures using the Okabe-Ito colour palette

The script operates on pre-computed permutation test outputs stored in structured confounder folders.
________________________________________
## Data Requirements

The script expects a CSV file containing:

- An  X-matrix with records oriented row-wise 
- A categorical class label
- A grouping variable (ID)
- At least one confounder column (confounder)

Rows with missing values for the confounder variable are removed.
________________________________________
## Preprocessing Steps

1.	Label encoding of the class variable
2.	Removal of metadata and non-numeric fields
3.	Median imputation for missing numeric values
4.	Removal of zero-variance features
5.	Extraction of:

    a.	X - numeric feature matrix

    b.	y - encoded labels

    c.	groups - individual IDs
________________________________________
## Cross-Validation Strategy

The pipeline uses StratifiedGroupKFold to ensure:

- Class balance in each fold
- No patient appears in both train and test sets


> ```
>   For each seed:
>       For each fold:
>           Fit PCA on training data
>           For j = 1 → n_components:
>               Train SVM on first j PCs
>               Evaluate on test set
>               Log metrics
>               Run permutation test
> 
> ``` 
________________________________________
## PCA-SVM Classification

For each fold and each PCA dimensionality:

1. PCA is fitted on training data only
2. Reduced features are passed to an SVM
3. Hyperparameters are tuned using GridSearchCV with:

   a.	Kernels: rbf, poly, linear, sigmoid

   b.   C values: 0.1 → 1

   c.	Gamma: scale, auto

4. Best model is evaluated on the held-out fold
5. Metrics logged:

    a. F1 (macro, micro, weighted)

    b. Accuracy

    c. Balanced accuracy

    d. Precision (macro)

    e. Recall (macro)
________________________________________
## Confounder-Restricted Permutation Test

This permutation test generates a null distribution of classification performances which represents the distribution of AUC values expected if the only signal present linking X to y is from latent influence of the confounder variable and/or noise.

Procedure:

For each fold and PCA dimension:

1. Shuffle the confounder column independently in train and test sets
2. Permute labels within each confounder level
3. Train PCA-SVM on permuted data
4. Compute AUC
5. Repeat for nperm = len(test_idx) permutations
6. Compute p-value

________________________________________
## Output Files
All outputs are written to ./outputs/.

1. foldwise_metrics.csv

One row per seed × fold × PCA component:

- PCA dimensionality
- Best SVM hyperparameters
- Full evaluation metrics
- Training CV score
- Skip reason (if any)

2. per_sample_predictions.csv

One row per test sample:

- True and predicted labels
- Probabilities for each class
- PCA dimensionality
- SVM hyperparameters

3. perm_null_full.csv

Full null distribution for each permutation test.

4. perm_test_results.csv

Summary per fold × PCA component:
- Observed AUC
- Null mean
- Null SD
- p-value
________________________________________
## Runtime Controls

Key parameters:

- CV_SPLITS = 5
- N_SEEDS = 20
- ncomp = 50
- paramspace - SVM hyperparameter grid
- PROGRESS_EVERY = 5 - progress logging frequency
________________________________________
## Reproducibility

The pipeline uses:

- Global seeds for each outer loop
- Independent seeds for each permutation (NULL_OFFSET + seed*1000 + i)
- Deterministic PCA and SVM when DETERMINISTIC=True
________________________________________
## File Structure

```
project/
│
├── permutation_testing_code.py 
├── file.csv 
│
└── outputs/
    ├── foldwise_metrics.csv
    ├── per_sample_predictions.csv
    ├── perm_null_full.csv
    └── perm_test_results.csv
```
________________________________________


# 2. Kolmogorov-Smirnov (KS) testing


The script **permutation_testing_ks_analysis.py** implements Kolmogorov-Smirnov (KS) testing & Empirical Cumulative Distribution Function (ECDF) Analysis.

The workflow performs:

- Null distribution visualisation for 5 categorical confounders
- Observed AUC comparison against null
- KS tests 
- ECDF visualisation
- Automated multi-page PDF generation
- Okabe-Ito colour palette used for figures

The script permutation_testing_ks_analysis.py operates on pre-computed permutation test outputs stored in structured confounder folders.
________________________________________
## Directory Structure
```

base_dir/
│
├── 01
│   └── outputs/
│       ├── perm_null_full.csv
│       └── perm_test_results.csv
├── 02
│   └── outputs/
│       ├── perm_null_full.csv
│       └── perm_test_results.csv
...
├── 10
│   └── outputs/
│       ├── perm_null_full.csv
│       └── perm_test_results.csv
```
Each folder contains:
- perm_null_full.csv - full null distribution from permutation testing
- perm_test_results.csv - observed AUC values per PCA component

________________________________________
## KS Testing & ECDF Analysis

The script **permutation_testing_ks_analysis.py** performs Kolmogorov-Smirnov tests comparing observed AUC distributions to null distributions.

Key Features

- Computes KS statistic and p-value
- Uses α = 0.001 significance threshold
- Generates two-panel figure per confounder:
     (a) Histogram + KDE overlay
     (b) ECDF comparison
- Uses Okabe-Ito palette for colour-blind-safe visualisation
- Saves one PDF:
      ALL_KS_PLOTS.pdf

Figure Details

Panel (a): Histogram + KDE
- Null distribution (colour-mapped)
- Observed distribution (Okabe-Ito orange)
- Clean spines and improved tick styling

Panel (b): ECDF curves
- Step plots for null and observed distributions
- KS statistic and p-value annotation
- Significance label
________________________________________
## Colour Scheme

The Okabe-Ito palette is used for maximum clarity:
- Blue - confounder 01
- Vermillion - confounder 02
- Additional colours assigned cyclically for remaining confounders
- Observed AUC distribution uses Okabe-Ito orange.
________________________________________
## Outputs

- ALL_KS_PLOTS.pdf

All files are saved to the specified save_dir.
________________________________________
## Requirements

numpy

pandas

matplotlib

seaborn

scipy
________________________________________
## Summary

The script **permutation_testing_ks_analysis.py** provides a statistical assessment of confounder influence on model performance using a PCA-SVM model as an illustrative example. 
________________________________________
## References:

Chaibub Neto, E., Pratap, A., Perumal, T.M., Tummalacherla, M., Bot, B.M., Mangravite, L., Omberg, L., 2019a. 
A Permutation Approach to Assess Confounding in Machine Learning Applications for Digital Health, in: 
Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’19. 
Association for Computing Machinery, New York, NY, USA, pp. 54–64. https://doi.org/10.1145/3292500.3330903

Chaibub Neto, E., Pratap, A., Perumal, T.M., Tummalacherla, M., Snyder, P., Bot, B.M., Trister, A.D., Friend, S.H., Mangravite, L., Omberg, L., 2019b. 
Detecting the impact of subject characteristics on machine learning-based diagnostic applications. 
Npj Digit. Med. 2, 99. https://doi.org/10.1038/s41746-019-0178-x

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html

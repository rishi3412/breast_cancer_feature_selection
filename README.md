# Breast Cancer Classification Using Feature Selection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Wisconsin%20Diagnostic-green)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

A comparative machine-learning study exploring how different
**feature-selection techniques** affect breast-cancer classification
performance.

> **Academic project:** This repository is intended for educational and
> research purposes. It is not a medical diagnostic tool.

## Table of contents

-   [Overview](#overview)
-   [Research objective](#research-objective)
-   [Dataset](#dataset)
-   [Methods compared](#methods-compared)
-   [Project structure](#project-structure)
-   [Getting started](#getting-started)
-   [Running the experiments](#running-the-experiments)
-   [Evaluation and outputs](#evaluation-and-outputs)
-   [Experimental protocol](#experimental-protocol)
-   [Current reference run](#current-reference-run)
-   [Limitations](#limitations)
-   [References](#references)

## Overview

The project uses the **Wisconsin Breast Cancer Diagnostic (WDBC)**
dataset to classify observations as:

-   **Malignant** --- encoded as `1`
-   **Benign** --- encoded as `0`

It compares three feature-selection approaches:

1.  **Filter method:** ANOVA F-test (`SelectKBest`)
2.  **Wrapper method:** Recursive Feature Elimination (RFE)
3.  **Embedded method:** L1-regularized Logistic Regression

A Logistic Regression model trained with all available features is
included as a baseline. The comparison considers both predictive
performance and the number of features retained.

## Research objective

The central research question is:

**How do filter, wrapper, and embedded feature-selection techniques
affect the performance and feature count of machine-learning models for
breast-cancer classification?**

The experiments aim to:

-   Compare classification models using a consistent train/test split.
-   Measure the effect of reducing the original feature set.
-   Compare accuracy, precision, recall, F1-score, and ROC-AUC.
-   Identify which features each selection technique retains.
-   Discuss the trade-off between model simplicity and predictive
    performance.

## Dataset

**Name:** Wisconsin Breast Cancer Diagnostic (WDBC)

The included CSV contains:

  Property         Description
  ---------------- ---------------------------------
  Observations     569
  Input features   30 numeric features
  Target           `diagnosis`
  Classes          `malignant`, `benign`
  Missing values   None in the distributed dataset

The 30 measurements describe characteristics of cell nuclei computed
from digitized images of breast-mass samples. They include measurements
such as radius, texture, perimeter, area, smoothness, compactness,
concavity, and symmetry.

**Dataset sources:**

-   [UCI Machine Learning Repository --- Breast Cancer Wisconsin
    (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
-   [Scikit-learn --- `load_breast_cancer`
    documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

The file `data/breast_cancer_wisconsin.csv` is included so the
experiments can run offline after dependencies are installed.

## Methods compared

  ---------------------------------------------------------------------------
  Experiment       Feature-selection   Classifier           Selected features
                   technique                            
  ---------------- ------------------- ---------------- ---------------------
  Baseline         No feature          Logistic                        All 30
                   selection           Regression       

  Solution 1       ANOVA F-test        Logistic                            10
                   (`SelectKBest`)     Regression       

  Solution 2       Recursive Feature   Random Forest                       10
                   Elimination (RFE)                    

  Solution 3       L1-based embedded   Linear SVM           Determined by the
                   selection                                  fitted selector
  ---------------------------------------------------------------------------

### 1. ANOVA F-test --- Filter method

`SelectKBest` ranks features using the ANOVA F-statistic and retains the
ten highest-scoring features. Feature ranking is performed independently
of the final classifier.

### 2. Recursive Feature Elimination --- Wrapper method

RFE repeatedly fits an estimator and removes lower-ranked features until
ten remain. In this project, Logistic Regression is used as the RFE
estimator, followed by a Random Forest classifier on the selected
features.

### 3. L1 regularization --- Embedded method

An L1-regularized Logistic Regression model encourages some feature
coefficients to become zero. `SelectFromModel` uses the fitted
coefficients to retain the selected features. A linear Support Vector
Machine (SVM) then performs classification using that reduced feature
set.

## Project structure

``` text
breast_cancer_feature_selection_cca/
│
├── data/
│   └── breast_cancer_wisconsin.csv
│
├── src/
│   └── run_experiments.py
│
├── results/
│   ├── metrics_comparison.csv
│   ├── selected_features.csv
│   ├── model_summary.txt
│   ├── feature_selection_comparison.png
│   └── confusion_matrix_*.png
│
├── requirements.txt
├── METHODOLOGY.md
├── README.md
└── .gitignore
```

The `results/` files are generated by running the experiment script.
They are included in this project bundle as a reference run.

## Getting started

### Requirements

-   Python 3.10 or newer (Python 3.10 or 3.11 is recommended)
-   VS Code or another Python IDE
-   Git (optional, for publishing to GitHub)

### 1. Clone the repository

``` bash
git clone https://github.com/rishi3412/breast-cancer-feature-selection.git
cd breast-cancer-feature-selection
```

If you downloaded the ZIP instead, extract it and open the extracted
folder in VS Code.

### 2. Create a virtual environment

**Windows PowerShell:**

``` powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the experiments

From the repository root, run:

``` bash
python src/run_experiments.py
```

The script loads the local CSV, creates a stratified train/test split,
trains the baseline and three feature-selection pipelines, evaluates
each model, and writes the output files to `results/`.

No manual dataset download is required.

## Evaluation and outputs

The script reports:

  -----------------------------------------------------------------------
  Metric                              What it measures
  ----------------------------------- -----------------------------------
  Accuracy                            Overall proportion of correct
                                      predictions

  Precision (malignant)               Among predicted malignant cases,
                                      the proportion that are malignant

  Recall (malignant)                  Among actual malignant cases, the
                                      proportion detected

  F1-score (malignant)                Harmonic mean of malignant
                                      precision and recall

  ROC-AUC                             Ability to rank malignant cases
                                      above benign cases across
                                      thresholds

  Selected feature count              Number of input features retained

  Training time                       Time taken to fit the pipeline in
                                      the current run
  -----------------------------------------------------------------------

**Generated files:**

-   `metrics_comparison.csv` --- metric comparison for all four models.
-   `selected_features.csv` --- feature names retained by each
    experiment.
-   `model_summary.txt` --- text summary of the run and its metrics.
-   `feature_selection_comparison.png` --- chart comparing evaluation
    metrics.
-   `confusion_matrix_*.png` --- confusion matrix for each model.

## Experimental protocol

-   A stratified 80:20 train/test split is used.
-   `random_state=42` is set for reproducibility.
-   Malignant is encoded as the positive class (`1`); benign is encoded
    as `0`.
-   Imputation, scaling, feature selection, and classification are kept
    inside Scikit-learn pipelines.
-   The pipelines are fitted on training data only, then evaluated on
    the held-out test data.

Keeping preprocessing and feature selection inside the pipeline helps
prevent information from the test set leaking into model training.

For a stronger research-paper evaluation, consider repeated stratified
cross-validation, reporting mean and standard deviation, and testing
whether conclusions remain stable across different random seeds. Do not
tune the model using the held-out test set.

## Current reference run

The included implementation was executed with the fixed 80:20 split and
`random_state=42`. One reference run produced:

  Model                            Features retained   Accuracy
  ------------------------------ ------------------- ----------
  Baseline Logistic Regression                    30     97.37%
  ANOVA + Logistic Regression                     10     95.61%
  RFE + Random Forest                             10     97.37%
  L1 selection + Linear SVM                        7     95.61%

These values describe one run of this implementation; they are not
guaranteed results for every environment or software version. Run the
script yourself and use the metrics generated on your machine in your
report. Avoid claiming that one method is universally best based on a
single split.

## Limitations

-   The dataset is relatively small and comes from a specific
    diagnostic-data collection.
-   A single train/test split does not fully establish generalization.
-   Feature-selection results can vary with the estimator,
    hyperparameters, and data split.
-   High test accuracy on this dataset does not demonstrate clinical
    validity or suitability for patient care.

This project is an educational machine-learning comparison, not a
substitute for clinical assessment or medical diagnosis.

## References

1.  Wolberg, W. H., Street, W. N., & Mangasarian, O. L. *Breast Cancer
    Wisconsin (Diagnostic) Data Set*. UCI Machine Learning Repository.
    https://doi.org/10.24432/C5DW2B
2.  Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in
    Python. *Journal of Machine Learning Research*, 12, 2825--2830.
    https://jmlr.org/papers/v12/pedregosa11a.html
3.  Scikit-learn documentation: [Feature
    selection](https://scikit-learn.org/stable/modules/feature_selection.html),
    [Pipelines](https://scikit-learn.org/stable/modules/compose.html#pipeline),
    and [Breast cancer dataset
    loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).



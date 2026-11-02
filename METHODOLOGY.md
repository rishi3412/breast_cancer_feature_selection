# Methodology

## Research question
How do filter, wrapper, and embedded feature-selection methods affect the performance and feature count of breast-cancer classification models?

## Dataset and target
The Wisconsin Breast Cancer Diagnostic dataset contains 569 observations and 30 real-valued features. The target has two classes: benign and malignant. For this project, malignant is encoded as 1 and benign as 0.

## Experimental design
A stratified 80:20 train-test split is used with random_state=42. Each method is implemented in a scikit-learn Pipeline so that imputation, scaling, and feature selection are learned from training data only.

## Methods
- ANOVA F-test: a filter method that ranks each feature by its univariate association with the class label; the top 10 are retained.
- RFE: a wrapper method that recursively eliminates features using a logistic-regression estimator until 10 remain. A random forest then classifies the reduced feature set.
- L1 embedded selection: an L1-regularized logistic regression produces sparse coefficients; SelectFromModel retains non-zero-coefficient features. A linear SVM classifies the selected subset.

## Evaluation
Accuracy, malignant precision, malignant recall, malignant F1-score, ROC-AUC, selected-feature count, training time, and confusion matrices are reported. Malignant recall is particularly important to discuss because false negatives represent malignant cases predicted as benign.

## Limitations
A single hold-out split can vary with the random seed. This dataset is small and relatively clean. Feature selection does not establish clinical usefulness, causality, or generalization to real-world populations. A medical professional must not use this project as a diagnostic tool.

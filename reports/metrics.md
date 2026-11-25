# Model evaluation report

All numbers are on a held-out, stratified 20% test split that the models never saw (5-fold CV ROC-AUC on the training split also shown). Thresholds marked *tuned* were chosen on out-of-fold training predictions only.

## Data preparation

```json
{
  "pima": {
    "dataset": "diabetes.csv (Pima-style)",
    "rows_raw": 9054,
    "exact_duplicate_rows": 8276,
    "rows_after_dedup": 778,
    "disguised_missing_zeros": {
      "Glucose": 5,
      "BloodPressure": 36,
      "SkinThickness": 229,
      "Insulin": 377,
      "BMI": 11
    },
    "missing_after_cleaning": {
      "Pregnancies": 0,
      "Glucose": 5,
      "BloodPressure": 36,
      "SkinThickness": 229,
      "Insulin": 377,
      "BMI": 11,
      "DiabetesPedigreeFunction": 0,
      "Age": 0
    },
    "class_balance": {
      "0": 506,
      "1": 272
    }
  },
  "diabd": {
    "dataset": "DiaBD (Bangladesh)",
    "rows_raw": 5288,
    "exact_duplicate_rows": 0,
    "implausible_set_missing": {
      "height": 6,
      "weight": 4,
      "pulse_rate": 3,
      "systolic_bp": 1,
      "diastolic_bp": 0,
      "glucose": 23,
      "age": 0,
      "bp_diastolic_ge_systolic": 0,
      "bmi_recorded": 5
    },
    "bmi_recorded_vs_computed_mismatch_gt1": 0,
    "missing_after_cleaning": {
      "age": 0,
      "gender_male": 0,
      "pulse_rate": 3,
      "systolic_bp": 1,
      "diastolic_bp": 0,
      "glucose": 23,
      "bmi": 5,
      "family_diabetes": 0,
      "hypertensive": 0,
      "family_hypertension": 0,
      "cardiovascular_disease": 0,
      "stroke": 0
    },
    "class_balance": {
      "0": 4946,
      "1": 342
    },
    "hypertensive_balance": {
      "0": 4701,
      "1": 587
    }
  }
}
```

## Structured models - pima (train 622, test 156)

| Model | CV ROC-AUC | Test ROC-AUC | PR-AUC | Accuracy | Bal. acc | Precision | Recall | F1 (tuned thr) |
|---|---|---|---|---|---|---|---|---|
| Support Vector Machine **(mandatory)** | 0.835 ± 0.033 | 0.782 | 0.693 | 0.692 | 0.713 | 0.544 | 0.782 | 0.642 |
| Logistic Regression **(mandatory)** | 0.844 ± 0.039 | 0.778 | 0.616 | 0.673 | 0.694 | 0.525 | 0.764 | 0.622 |
| LDA | 0.845 ± 0.039 | 0.778 | 0.614 | 0.686 | 0.700 | 0.539 | 0.746 | 0.626 |
| Extra Trees | 0.857 ± 0.040 | 0.777 | 0.628 | 0.679 | 0.703 | 0.531 | 0.782 | 0.632 |
| Gaussian Naive Bayes | 0.811 ± 0.043 | 0.760 | 0.563 | 0.635 | 0.668 | 0.489 | 0.782 | 0.601 |
| MLP Neural Net | 0.774 ± 0.056 | 0.760 | 0.565 | 0.660 | 0.684 | 0.512 | 0.764 | 0.613 |
| Random Forest **(mandatory)** | 0.852 ± 0.042 | 0.746 | 0.620 | 0.673 | 0.665 | 0.530 | 0.636 | 0.579 |
| K-Nearest Neighbours | 0.832 ± 0.036 | 0.744 | 0.645 | 0.705 | 0.702 | 0.567 | 0.691 | 0.623 |
| Gradient Boosting | 0.850 ± 0.043 | 0.742 | 0.587 | 0.647 | 0.666 | 0.500 | 0.727 | 0.593 |
| AdaBoost | 0.843 ± 0.038 | 0.742 | 0.586 | 0.660 | 0.663 | 0.514 | 0.673 | 0.583 |
| HistGradientBoosting (native NaN) | 0.826 ± 0.036 | 0.736 | 0.608 | 0.641 | 0.661 | 0.494 | 0.727 | 0.588 |
| Decision Tree **(mandatory)** | 0.777 ± 0.036 | 0.735 | 0.566 | 0.660 | 0.680 | 0.512 | 0.746 | 0.607 |
| XGBoost **(mandatory)** | 0.820 ± 0.035 | 0.730 | 0.581 | 0.615 | 0.628 | 0.468 | 0.673 | 0.552 |

## Structured models - diabd (train 4230, test 1058)

| Model | CV ROC-AUC | Test ROC-AUC | PR-AUC | Accuracy | Bal. acc | Precision | Recall | F1 (tuned thr) |
|---|---|---|---|---|---|---|---|---|
| XGBoost **(mandatory)** | 0.869 ± 0.032 | 0.855 | 0.354 | 0.908 | 0.759 | 0.367 | 0.588 | 0.452 |
| Random Forest **(mandatory)** | 0.882 ± 0.043 | 0.852 | 0.346 | 0.903 | 0.681 | 0.312 | 0.426 | 0.360 |
| Gradient Boosting | 0.882 ± 0.037 | 0.850 | 0.364 | 0.905 | 0.758 | 0.357 | 0.588 | 0.444 |
| HistGradientBoosting (native NaN) | 0.862 ± 0.035 | 0.847 | 0.301 | 0.917 | 0.654 | 0.353 | 0.353 | 0.353 |
| Extra Trees | 0.875 ± 0.052 | 0.843 | 0.340 | 0.878 | 0.723 | 0.274 | 0.544 | 0.364 |
| Support Vector Machine **(mandatory)** | 0.858 ± 0.062 | 0.841 | 0.349 | 0.882 | 0.725 | 0.282 | 0.544 | 0.372 |
| AdaBoost | 0.886 ± 0.031 | 0.840 | 0.348 | 0.911 | 0.692 | 0.349 | 0.441 | 0.390 |
| Logistic Regression **(mandatory)** | 0.869 ± 0.051 | 0.827 | 0.279 | 0.917 | 0.647 | 0.348 | 0.338 | 0.343 |
| Decision Tree **(mandatory)** | 0.823 ± 0.039 | 0.827 | 0.369 | 0.863 | 0.769 | 0.270 | 0.662 | 0.383 |
| Gaussian Naive Bayes | 0.801 ± 0.083 | 0.816 | 0.231 | 0.073 | 0.504 | 0.065 | 1.000 | 0.122 |
| K-Nearest Neighbours | 0.828 ± 0.067 | 0.813 | 0.294 | 0.909 | 0.705 | 0.348 | 0.471 | 0.400 |
| MLP Neural Net | 0.836 ± 0.050 | 0.808 | 0.271 | 0.927 | 0.660 | 0.421 | 0.353 | 0.384 |
| LDA | 0.862 ± 0.064 | 0.801 | 0.289 | 0.927 | 0.694 | 0.433 | 0.426 | 0.430 |

## Structured models - general (train 4852, test 1214)

| Model | CV ROC-AUC | Test ROC-AUC | PR-AUC | Accuracy | Bal. acc | Precision | Recall | F1 (tuned thr) |
|---|---|---|---|---|---|---|---|---|
| AdaBoost | 0.893 ± 0.009 | 0.885 | 0.559 | 0.915 | 0.751 | 0.588 | 0.545 | 0.565 |
| Gradient Boosting | 0.898 ± 0.010 | 0.882 | 0.516 | 0.909 | 0.776 | 0.547 | 0.610 | 0.577 |
| Random Forest **(mandatory)** | 0.898 ± 0.010 | 0.880 | 0.531 | 0.913 | 0.735 | 0.578 | 0.512 | 0.543 |
| Support Vector Machine **(mandatory)** | 0.888 ± 0.014 | 0.879 | 0.553 | 0.887 | 0.764 | 0.457 | 0.610 | 0.523 |
| Extra Trees | 0.896 ± 0.015 | 0.877 | 0.531 | 0.913 | 0.731 | 0.579 | 0.504 | 0.539 |
| MLP Neural Net | 0.890 ± 0.012 | 0.876 | 0.557 | 0.913 | 0.728 | 0.581 | 0.496 | 0.535 |
| Logistic Regression **(mandatory)** | 0.888 ± 0.014 | 0.875 | 0.495 | 0.898 | 0.781 | 0.497 | 0.634 | 0.557 |
| LDA | 0.885 ± 0.013 | 0.874 | 0.505 | 0.900 | 0.764 | 0.507 | 0.594 | 0.547 |
| XGBoost **(mandatory)** | 0.902 ± 0.006 | 0.864 | 0.568 | 0.900 | 0.746 | 0.507 | 0.553 | 0.529 |
| Decision Tree **(mandatory)** | 0.849 ± 0.020 | 0.863 | 0.487 | 0.892 | 0.792 | 0.477 | 0.667 | 0.556 |
| Gaussian Naive Bayes | 0.841 ± 0.022 | 0.858 | 0.380 | 0.870 | 0.765 | 0.408 | 0.634 | 0.497 |
| HistGradientBoosting (native NaN) | 0.891 ± 0.008 | 0.857 | 0.539 | 0.903 | 0.744 | 0.519 | 0.545 | 0.532 |
| K-Nearest Neighbours | 0.869 ± 0.020 | 0.846 | 0.492 | 0.904 | 0.720 | 0.531 | 0.488 | 0.508 |

## Missing-value strategy comparison (Pima, 5-fold CV ROC-AUC)

| Strategy | ROC-AUC | Rows |
|---|---|---|
| Complete-case (drop rows) | Logistic Regression | 0.846 ± 0.042 | only 397/778 rows usable |
| Complete-case (drop rows) | Random Forest | 0.845 ± 0.032 | only 397/778 rows usable |
| Median fill | Logistic Regression | 0.831 ± 0.028 | all 778 rows |
| Median fill | Random Forest | 0.834 ± 0.020 | all 778 rows |
| KNN imputation | Logistic Regression | 0.832 ± 0.026 | all 778 rows |
| KNN imputation | Random Forest | 0.832 ± 0.020 | all 778 rows |
| MICE + missing indicators | Logistic Regression | 0.826 ± 0.036 | all 778 rows |
| MICE + missing indicators | Random Forest | 0.834 ± 0.027 | all 778 rows |
| XGBoost native missing handling | XGBoost | 0.794 ± 0.032 | all 778 rows |

## Robustness: test ROC-AUC when measurement cells are blanked at random (general model)

| Missing | Logistic Regression | Decision Tree | Random Forest | XGBoost | Support Vector Machine |
|---|---|---|---|---|---|
| 0% cells missing | 0.875 | 0.863 | 0.880 | 0.864 | 0.879 |
| 10% cells missing | 0.858 | 0.844 | 0.880 | 0.846 | 0.844 |
| 20% cells missing | 0.841 | 0.843 | 0.868 | 0.830 | 0.793 |
| 30% cells missing | 0.824 | 0.814 | 0.846 | 0.802 | 0.772 |
| 50% cells missing | 0.806 | 0.808 | 0.829 | 0.778 | 0.645 |

## NLP baseline, BioBERT and Hybrid XGBoost (DiaBD + clinical notes)

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | Specificity | F1 |
|---|---|---|---|---|---|---|---|
| TF-IDF + SVM (text only) | 0.729 | 0.174 | 0.895 | 0.274 | 0.382 | 0.930 | 0.319 |
| BioBERT embeddings + Logistic Regression (text only) | 0.709 | 0.158 | 0.880 | 0.196 | 0.279 | 0.921 | 0.230 |
| Hybrid XGBoost (structured + NLP-extracted + BioBERT) | 0.823 | 0.340 | 0.891 | 0.306 | 0.544 | 0.915 | 0.392 |
| Ablation: XGBoost measurements only | 0.831 | 0.262 | 0.887 | 0.276 | 0.471 | 0.915 | 0.348 |

## Hypertension assessment model (DiaBD)

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | Specificity | F1 |
|---|---|---|---|---|---|---|---|
| XGBoost | 0.861 | 0.446 | 0.807 | 0.338 | 0.778 | 0.811 | 0.471 |
| Logistic Regression | 0.852 | 0.427 | 0.841 | 0.376 | 0.658 | 0.864 | 0.478 |
| Random Forest | 0.855 | 0.423 | 0.825 | 0.356 | 0.718 | 0.839 | 0.476 |

## Clinical NLP extraction accuracy (rule-based extractor vs recorded attributes)

| Attribute | Accuracy | F1 |
|---|---|---|
| hypertensive | 1.000 | 1.000 |
| cardiovascular_disease | 1.000 | 1.000 |
| stroke | 1.000 | 1.000 |
| family_diabetes | 1.000 | 1.000 |
| family_hypertension | 1.000 | 1.000 |
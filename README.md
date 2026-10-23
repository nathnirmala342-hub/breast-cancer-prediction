# Breast Cancer Prediction using Machine Learning

A beginner-friendly data science project that predicts whether a breast tumour is **malignant** (cancerous) or **benign** (not cancerous) from measurements of cell nuclei.

> **Note:** This is a learning project, not a medical tool. It must not be used to diagnose anyone.

## Problem

Doctors examine cells from a breast mass under a microscope. From a digitised image, 10 measurements (such as radius, texture, smoothness, and concavity) are computed for each cell nucleus, and the mean, error, and "worst" (largest) value of each are recorded, giving 30 features per patient.

**Question:** Can we predict from these 30 numbers whether a tumour is malignant or benign?

## Dataset

- **Breast Cancer Wisconsin (Diagnostic)**, from the UCI Machine Learning Repository
- Loaded directly from scikit-learn (`sklearn.datasets.load_breast_cancer`), so no download is needed
- 569 patients, 30 numeric features, no missing values
- 357 benign and 212 malignant cases (about 63% / 37%)

I relabelled the target so that **1 = malignant**, because malignant is the case we most want to catch.

## Method

1. Looked at the class balance and how key features differ between the two classes
2. Split the data 80% train / 20% test, **stratified** so both sets keep the same benign/malignant ratio
3. Compared three models using **5-fold cross-validation on the training set only**:
   - Logistic Regression (with feature scaling)
   - Decision Tree (max depth 4)
   - Random Forest (300 trees)
4. Chose the best model by cross-validation, then tested it once on the untouched test set
5. Checked which features matter most

Scaling is done inside a pipeline so the test data never influences training (this avoids data leakage).

## Results

Cross-validation accuracy (training set):

| Model | CV accuracy |
|---|---|
| Logistic Regression | 0.974 |
| Random Forest | 0.960 |
| Decision Tree | 0.921 |

Best model: **Logistic Regression**. Test set results (114 patients it had never seen):

| Metric | Value |
|---|---|
| Accuracy | 96.5% |
| Precision (malignant) | 97.5% |
| Recall (malignant) | 92.9% |
| ROC-AUC | 0.996 |

Confusion matrix: 71 benign correctly identified, 39 malignant correctly identified, 1 benign wrongly flagged as malignant, and **3 malignant cases missed**.

### Key findings

- The most important features were **worst perimeter, worst area, worst concave points, mean concave points, and worst radius**. In short, larger and more irregularly shaped nuclei point towards malignancy.
- A simple model (Logistic Regression) beat more complex ones here, which happens when the classes are well separated.
- In a real medical setting, **missing a malignant case (false negative) is worse than a false alarm**. Here 3 of 42 malignant cases were missed, so recall is the number to improve first.

## Limitations

- The test set is small (114 patients), so the exact numbers could change with a different split. The cross-validation figures give a more stable picture.
- The dataset is old and fairly clean. Real hospital data is messier and would need external validation.
- I did not tune hyperparameters or adjust the decision threshold.

## Ideas for improvement

- Lower the decision threshold to catch more malignant cases, and measure the cost in extra false alarms
- Tune hyperparameters with `GridSearchCV`
- Try other models such as SVM or gradient boosting
- Explain individual predictions with SHAP

## Project structure

```
breast_cancer_prediction/
├── predict_breast_cancer.py   # full analysis, run this file
├── requirements.txt
├── README.md
└── outputs/
    ├── 01_class_balance.png
    ├── 02_feature_distributions.png
    ├── 03_model_comparison.png
    ├── 04_confusion_matrix.png
    ├── 05_roc_curves.png
    ├── 06_feature_importance.png
    ├── metrics.json
    └── run_log.txt
```

## How to run

```bash
pip install -r requirements.txt
python predict_breast_cancer.py
```

Charts and metrics are saved to the `outputs/` folder.

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib

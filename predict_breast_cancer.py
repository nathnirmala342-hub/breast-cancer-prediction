"""
Breast Cancer Prediction (Wisconsin Diagnostic dataset)

Goal: predict whether a tumour is malignant or benign from measurements of
cell nuclei taken from a digitised image of a breast mass.

Dataset: built into scikit-learn (load_breast_cancer), 569 samples, 30 features.
Original source: UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic).

Steps:
  1. Load and inspect the data
  2. Exploratory data analysis (class balance, feature differences)
  3. Train/test split (stratified) and 5-fold cross-validation
  4. Compare three models: Logistic Regression, Decision Tree, Random Forest
  5. Evaluate the best model: confusion matrix, precision, recall, ROC-AUC
  6. Look at which features matter most

Run:  python predict_breast_cancer.py
Outputs are saved in the 'outputs' folder.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, we only save images
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

# Two clearly different colours that stay readable for colour-blind viewers
BENIGN = "#2a6fbb"     # blue
MALIGNANT = "#d9731a"  # orange
GREY = "#6b7280"

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})


# ---------------------------------------------------------------- 1. load data
raw = load_breast_cancer(as_frame=True)
X = raw.data.copy()
# In scikit-learn, 0 = malignant and 1 = benign. We flip it so that
# 1 = malignant, because malignant is the case we want to catch ("positive").
y = 1 - raw.target
y.name = "malignant"

print("Samples:", X.shape[0], "| Features:", X.shape[1])
print("Missing values:", int(X.isna().sum().sum()))
print("Benign (0):", int((y == 0).sum()), "| Malignant (1):", int((y == 1).sum()))


# ------------------------------------------------------------------- 2. EDA
# 2a. Class balance
fig, ax = plt.subplots(figsize=(4.5, 3.6))
counts = y.value_counts().sort_index()
bars = ax.bar(["Benign", "Malignant"], counts.values, color=[BENIGN, MALIGNANT], width=0.55)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 5, str(v), ha="center", fontsize=10)
ax.set_ylabel("Number of patients")
ax.set_title("Class balance in the dataset")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(OUT / "01_class_balance.png")
plt.close(fig)

# 2b. How do the most useful features differ between the two classes?
eda_features = ["worst radius", "worst concave points", "mean texture", "mean smoothness"]
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
for ax, col in zip(axes.ravel(), eda_features):
    ax.hist(X.loc[y == 0, col], bins=25, alpha=0.75, color=BENIGN, label="Benign")
    ax.hist(X.loc[y == 1, col], bins=25, alpha=0.75, color=MALIGNANT, label="Malignant")
    ax.set_title(col)
    ax.set_ylabel("Count")
axes[0, 0].legend(frameon=False)
fig.suptitle("Malignant tumours tend to have larger, more irregular nuclei", y=1.0)
fig.tight_layout()
fig.savefig(OUT / "02_feature_distributions.png")
plt.close(fig)


# ----------------------------------------------- 3. split data and set up CV
# stratify keeps the benign/malignant ratio the same in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print("\nTrain size:", len(X_train), "| Test size:", len(X_test))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Scaling is done INSIDE the pipeline so that the test data never influences it
# (this avoids "data leakage").
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    ),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
}


# ------------------------------------------------------------ 4. compare models
cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    cv_results[name] = (scores.mean(), scores.std())
    print(f"{name:20s} CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# A dot plot (not bars) because the axis does not start at zero: bars would
# exaggerate the differences. Each dot is the mean, the line is +/- 1 std.
fig, ax = plt.subplots(figsize=(6, 3.6))
names = list(cv_results)
means = [cv_results[n][0] for n in names]
stds = [cv_results[n][1] for n in names]
x_pos = np.arange(len(names))
ax.errorbar(x_pos, means, yerr=stds, fmt="o", color=GREY, ecolor=GREY,
            elinewidth=2, capsize=5, markersize=9)
for x, m in zip(x_pos, means):
    ax.text(x + 0.12, m, f"{m:.3f}", va="center", fontsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(names)
ax.set_xlim(-0.5, len(names) - 0.5)
ax.set_ylim(0.86, 1.0)
ax.set_ylabel("5-fold CV accuracy")
ax.set_title("Model comparison (mean +/- 1 std across folds)")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(OUT / "03_model_comparison.png")
plt.close(fig)

# Pick the best model by cross-validation (NOT by test score, to keep the test set honest)
best_name = max(cv_results, key=lambda n: cv_results[n][0])
best_model = models[best_name]
print("\nBest model by cross-validation:", best_name)


# ---------------------------------------------------- 5. evaluate on test set
best_model.fit(X_train, y_train)
pred = best_model.predict(X_test)
proba = best_model.predict_proba(X_test)[:, 1]

metrics = {
    "best_model": best_name,
    "test_accuracy": round(accuracy_score(y_test, pred), 4),
    "test_precision_malignant": round(precision_score(y_test, pred), 4),
    "test_recall_malignant": round(recall_score(y_test, pred), 4),
    "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
    "cv_accuracy": {n: round(v[0], 4) for n, v in cv_results.items()},
}
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
metrics["confusion"] = {"true_benign": int(tn), "false_malignant": int(fp),
                        "missed_malignant": int(fn), "true_malignant": int(tp)}

print("\nTest set results:")
print(classification_report(y_test, pred, target_names=["Benign", "Malignant"]))
print("ROC-AUC:", metrics["test_roc_auc"])
print("Confusion:", metrics["confusion"])

with open(OUT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Confusion matrix
fig, ax = plt.subplots(figsize=(4.6, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=["Benign", "Malignant"], cmap="Blues", ax=ax, colorbar=False
)
ax.grid(False)
ax.set_title(f"Confusion matrix: {best_name}")
fig.tight_layout()
fig.savefig(OUT / "04_confusion_matrix.png")
plt.close(fig)

# ROC curves for all models on the test set
fig, ax = plt.subplots(figsize=(5, 4.3))
palette = {"Logistic Regression": "#2a6fbb", "Decision Tree": "#9ca3af", "Random Forest": "#d9731a"}
for name, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, p)
    ax.plot(fpr, tpr, lw=2, color=palette[name], label=f"{name} (AUC {roc_auc_score(y_test, p):.3f})")
ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate (recall)")
ax.set_title("ROC curves on the test set")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "05_roc_curves.png")
plt.close(fig)

# Refit best model (the loop above refit every model on the same training data,
# so this is the same fit; done again to be explicit)
best_model.fit(X_train, y_train)


# ----------------------------------------------------- 6. feature importance
# Use a Random Forest to rank features, whatever the best model was.
rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE).fit(X_train, y_train)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.barh(imp.index[::-1], imp.values[::-1], color=GREY)
ax.set_xlabel("Importance (Random Forest)")
ax.set_title("Top 10 features for predicting malignancy")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(OUT / "06_feature_importance.png")
plt.close(fig)

print("\nTop 5 features:")
print(imp.head(5).round(3).to_string())
metrics["top_features"] = {k: round(float(v), 4) for k, v in imp.head(5).items()}
with open(OUT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nDone. Charts and metrics saved to:", OUT)

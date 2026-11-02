"""
CCA project: Breast Cancer Classification with Feature Selection.

Run from the project root:
    python src/run_experiments.py

Outputs:
    results/metrics_comparison.csv
    results/selected_features.csv
    results/confusion_matrix_*.png
    results/feature_selection_comparison.png
    results/feature_importance_*.png
    results/model_summary.txt

The dataset labels are "malignant" and "benign".
Positive class for clinical-style recall reporting is malignant.
All preprocessing and feature selection are fitted inside pipelines on training folds only.
"""
from pathlib import Path
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "breast_cancer_wisconsin.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FEATURES = 10

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place breast_cancer_wisconsin.csv in the data folder."
        )
    df = pd.read_csv(DATA_PATH)
    if "diagnosis" not in df.columns:
        raise ValueError("CSV must contain a 'diagnosis' target column.")
    X = df.drop(columns=["diagnosis"])
    y_text = df["diagnosis"].astype(str).str.lower().str.strip()
    valid = {"malignant", "benign"}
    if not set(y_text.unique()).issubset(valid):
        raise ValueError(f"Expected diagnosis labels {valid}; got {set(y_text.unique())}")
    # Encode malignant as 1 so malignant recall is explicit and clinically interpretable.
    y = (y_text == "malignant").astype(int)
    return X, y

def build_models():
    # Scaling is performed inside each pipeline, after splitting.
    baseline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced",
                                          random_state=RANDOM_STATE))
    ])

    anova = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=N_FEATURES)),
        ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced",
                                          random_state=RANDOM_STATE))
    ])

    rfe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("select", RFE(
            estimator=LogisticRegression(max_iter=5000, class_weight="balanced",
                                         random_state=RANDOM_STATE),
            n_features_to_select=N_FEATURES,
            step=1
        )),
        ("classifier", RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])

    # L1-regularized linear SVM selects a sparse subset of features.
    l1_svm = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("select", SelectFromModel(
            LogisticRegression(
                penalty="l1", solver="liblinear", C=0.1,
                class_weight="balanced", max_iter=5000,
                random_state=RANDOM_STATE
            ),
            threshold=1e-8
        )),
        ("classifier", SVC(
            kernel="linear", C=1.0, class_weight="balanced",
            probability=True, random_state=RANDOM_STATE
        ))
    ])
    return {
        "Baseline_All_30": baseline,
        "Solution_1_ANOVA_LogisticRegression": anova,
        "Solution_2_RFE_RandomForest": rfe,
        "Solution_3_L1_Selection_SVM": l1_svm
    }

def selected_feature_names(model, feature_names):
    selector = model.named_steps.get("select")
    if selector is None:
        return list(feature_names)
    mask = selector.get_support()
    return list(np.asarray(feature_names)[mask])

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    models = build_models()
    rows = []
    selected_rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        feats = selected_feature_names(model, X.columns)

        rows.append({
            "experiment": name,
            "features_selected": len(feats),
            "accuracy": accuracy_score(y_test, pred),
            "precision_malignant": precision_score(y_test, pred, zero_division=0),
            "recall_malignant": recall_score(y_test, pred, zero_division=0),
            "f1_malignant": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
            "training_time_seconds": elapsed
        })
        for f in feats:
            selected_rows.append({"experiment": name, "selected_feature": f})

        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Benign", "Malignant"],
                    yticklabels=["Benign", "Malignant"])
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(name.replace("_", " "))
        plt.tight_layout()
        plt.savefig(OUT / f"confusion_matrix_{name}.png", dpi=160)
        plt.close()

        print(f"\n{name}")
        print(f"Selected {len(feats)} of {X.shape[1]} features")
        print("Selected features:", ", ".join(feats))
        print(classification_report(
            y_test, pred, labels=[0, 1],
            target_names=["benign", "malignant"], zero_division=0
        ))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics_comparison.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(OUT / "selected_features.csv", index=False)

    plot_df = metrics.melt(
        id_vars="experiment",
        value_vars=["accuracy", "precision_malignant", "recall_malignant", "f1_malignant", "roc_auc"],
        var_name="metric", value_name="score"
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="experiment", y="score", hue="metric")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=18, ha="right")
    plt.ylabel("Score")
    plt.xlabel("")
    plt.title("Breast Cancer Classification: Model Comparison")
    plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT / "feature_selection_comparison.png", dpi=180)
    plt.close()

    summary = [
        "BREAST CANCER FEATURE SELECTION - EXPERIMENT SUMMARY",
        f"Dataset rows: {len(X)}",
        f"Original features: {X.shape[1]}",
        f"Train rows: {len(X_train)} | Test rows: {len(X_test)}",
        "Target encoding: malignant=1, benign=0",
        "Split: stratified 80/20, random_state=42",
        "Feature selection is fitted only on training data.",
        "",
        metrics.to_string(index=False)
    ]
    (OUT / "model_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\nSaved results to:", OUT)
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()

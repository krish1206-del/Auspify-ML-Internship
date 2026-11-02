"""Train and evaluate the Netflix Audience Rating Classification models.

Run from this directory with:
    python task3_rating_classifier.py
"""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
MIN_CLASS_COUNT = 50
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "Dataset.csv"
OUTPUT_DIR = BASE / "outputs"
SCREENSHOT_DIR = BASE / "screenshots"
OUTPUT_DIR.mkdir(exist_ok=True)
SCREENSHOT_DIR.mkdir(exist_ok=True)


def main() -> None:
    warnings.filterwarnings("ignore")
    df = pd.read_csv(DATA_PATH).drop_duplicates().copy()
    text_columns = ["type", "title", "director", "country", "rating", "duration", "listed_in"]
    for column in text_columns:
        df[column] = df[column].fillna("Unknown").astype(str).str.strip()

    original_counts = df["rating"].value_counts()
    rare_labels = original_counts[original_counts < MIN_CLASS_COUNT].index.tolist()
    df["rating_category"] = df["rating"].where(~df["rating"].isin(rare_labels), "Other (rare)")
    df["metadata_text"] = (
        df[["director", "country", "listed_in"]]
        .fillna("Unknown").astype(str).agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )

    feature_columns = ["metadata_text", "type", "duration", "release_year"]
    X, y = df[feature_columns].copy(), df["rating_category"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = ColumnTransformer(transformers=[
        ("text", TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=2,
            max_features=25000, sublinear_tf=True
        ), "metadata_text"),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["type", "duration"]),
        ("numeric", StandardScaler(), ["release_year"]),
    ])

    candidates = {
        "Most Frequent Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (unweighted)": LogisticRegression(max_iter=1800, solver="lbfgs", C=1.0),
        "Logistic Regression (balanced)": LogisticRegression(max_iter=1800, solver="lbfgs", C=1.0, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", max_depth=20, min_samples_leaf=3
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=160, random_state=RANDOM_STATE, class_weight="balanced_subsample",
            max_depth=30, min_samples_leaf=2, n_jobs=-1
        ),
    }
    fitted, preds, rows = {}, {}, []
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name], preds[name] = pipe, pred
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
            "Macro F1": f1_score(y_test, pred, average="macro"),
            "Weighted F1": f1_score(y_test, pred, average="weighted"),
        })

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    lr_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1800, solver="lbfgs")),
    ])
    lr_search = GridSearchCV(
        lr_pipe,
        {"model__C": [0.3, 1.0, 3.0], "model__class_weight": [None, "balanced"]},
        scoring="f1_macro", cv=cv, n_jobs=-1, refit=True,
    ).fit(X_train, y_train)

    tree_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])
    tree_search = GridSearchCV(
        tree_pipe,
        {
            "model__max_depth": [12, 20, None],
            "model__min_samples_leaf": [2, 5],
            "model__class_weight": ["balanced"],
        },
        scoring="f1_macro", cv=cv, n_jobs=-1, refit=True,
    ).fit(X_train, y_train)

    tuned = {
        "Tuned Logistic Regression": (lr_search.best_estimator_, lr_search.best_score_),
        "Tuned Decision Tree": (tree_search.best_estimator_, tree_search.best_score_),
    }
    for name, (model, cv_score) in tuned.items():
        pred = model.predict(X_test)
        fitted[name], preds[name] = model, pred
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
            "Macro F1": f1_score(y_test, pred, average="macro"),
            "Weighted F1": f1_score(y_test, pred, average="weighted"),
            "CV Macro F1": cv_score,
        })

    comparison = pd.DataFrame(rows).sort_values(["Macro F1", "Balanced Accuracy"], ascending=False)
    final_name = "Tuned Logistic Regression" if lr_search.best_score_ >= tree_search.best_score_ else "Tuned Decision Tree"
    final_model = fitted[final_name]
    final_pred = preds[final_name]
    metrics = {
        "Selected model": final_name,
        "Accuracy": float(accuracy_score(y_test, final_pred)),
        "Balanced accuracy": float(balanced_accuracy_score(y_test, final_pred)),
        "Macro F1": float(f1_score(y_test, final_pred, average="macro")),
        "Weighted F1": float(f1_score(y_test, final_pred, average="weighted")),
    }

    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(classification_report(y_test, final_pred, output_dict=True, zero_division=0)).T.to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, final_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(OUTPUT_DIR / "confusion_matrix.csv")
    pd.Series(y.value_counts(), name="title_count").to_csv(OUTPUT_DIR / "rating_category_counts.csv")
    sample = X_test.copy()
    sample["Actual rating"] = y_test
    sample["Predicted rating"] = final_pred
    sample["Correct"] = sample["Actual rating"] == sample["Predicted rating"]
    sample.drop(columns=["metadata_text"]).head(20).to_csv(OUTPUT_DIR / "sample_test_predictions.csv")
    joblib.dump(final_model, OUTPUT_DIR / "best_rating_classifier.joblib")
    summary = {
        **metrics,
        "dataset_rows": int(len(df)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "original_rating_categories": int(df["rating"].nunique()),
        "modelled_rating_categories": int(y.nunique()), "rare_labels_grouped": rare_labels,
        "rare_label_threshold": MIN_CLASS_COUNT,
        "logistic_cv_macro_f1": float(lr_search.best_score_),
        "tree_cv_macro_f1": float(tree_search.best_score_),
        "feature_columns": feature_columns, "random_state": RANDOM_STATE,
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("Model comparison:")
    print(comparison.round(4).to_string(index=False))
    print("\nSelected model:", final_name)
    print(json.dumps(metrics, indent=2))
    print("\nOutputs saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()

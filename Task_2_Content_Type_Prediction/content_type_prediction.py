"""Netflix Content Type Prediction - Auspify Technologies ML Internship, Task 2.

Predicts whether a Netflix catalogue entry is a Movie or TV Show from
metadata. The target column (`type`) and leakage-prone identifiers/title/
duration fields are excluded from predictors.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET = "type"
NUMERIC_FEATURES = ["release_year", "date_added_year", "date_added_month"]
CATEGORICAL_FEATURES = ["rating"]
TEXT_FEATURES = ["director", "country", "listed_in"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES


def split_comma_terms(value: Any) -> list[str]:
    """Turn comma-separated metadata into normalized multi-value tokens."""
    if pd.isna(value):
        return []
    return [re.sub(r"\s+", " ", part.strip().lower()) for part in str(value).split(",") if part.strip()]


def prepare_dataset(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Read the CSV and construct leakage-aware model features."""
    df = pd.read_csv(csv_path)
    required = {TARGET, "release_year", "date_added", "rating", "director", "country", "listed_in"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    df = df.drop_duplicates().copy()
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df[df[TARGET].isin(["Movie", "TV Show"])].copy()

    date_added = pd.to_datetime(df["date_added"], errors="coerce")
    features = pd.DataFrame(index=df.index)
    features["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    features["date_added_year"] = date_added.dt.year
    features["date_added_month"] = date_added.dt.month
    for column in ["rating", "director", "country", "listed_in"]:
        features[column] = df[column].fillna("Unknown").astype(str)

    # Deliberately do not use duration: values such as "90 min" and "2 Seasons"
    # practically disclose the target class and would create target leakage.
    y = df[TARGET].copy()
    return features[FEATURE_COLUMNS], y, df


def build_preprocessor(include_genres: bool = True) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    rating_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers = [
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("rating", rating_pipeline, CATEGORICAL_FEATURES),
        ("director_tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=3000), "director"),
        ("country_tokens", TfidfVectorizer(analyzer=split_comma_terms, min_df=1, max_features=500), "country"),
    ]
    if include_genres:
        transformers.append(
            ("genre_tokens", TfidfVectorizer(analyzer=split_comma_terms, min_df=1, max_features=500), "listed_in")
        )
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )


def build_models() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0, solver="liblinear", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=24,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, np.ndarray]]:
    models = build_models()
    fitted: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}
    rows: list[dict[str, float | str]] = []

    baseline = Pipeline(
        steps=[("preprocessor", build_preprocessor()), ("classifier", DummyClassifier(strategy="most_frequent"))]
    )
    baseline.fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    rows.append(
        {
            "Model": "Majority-class baseline",
            "Accuracy": accuracy_score(y_test, base_pred),
            "Precision (macro)": precision_score(y_test, base_pred, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_test, base_pred, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_test, base_pred, average="macro", zero_division=0),
        }
    )
    predictions["Majority-class baseline"] = base_pred

    for name, classifier in models.items():
        pipeline = Pipeline(steps=[("preprocessor", build_preprocessor()), ("classifier", classifier)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        fitted[name] = pipeline
        predictions[name] = pred
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision (macro)": precision_score(y_test, pred, average="macro", zero_division=0),
                "Recall (macro)": recall_score(y_test, pred, average="macro", zero_division=0),
                "F1 (macro)": f1_score(y_test, pred, average="macro", zero_division=0),
            }
        )

    results = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
    return results, fitted, predictions


def create_visuals(
    full_df: pd.DataFrame,
    results: pd.DataFrame,
    best_name: str,
    y_test: pd.Series,
    best_predictions: np.ndarray,
    output_dir: str | Path,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Target distribution
    counts = full_df[TARGET].value_counts().reindex(["Movie", "TV Show"])
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Netflix Content Type Distribution")
    ax.set_xlabel("Content type")
    ax.set_ylabel("Number of titles")
    ax.tick_params(axis="x", rotation=0)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%d")
    fig.tight_layout()
    fig.savefig(out / "content_type_distribution.png", dpi=180)
    plt.close(fig)

    # Model comparison
    chart_df = results.sort_values("Accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(chart_df["Model"], chart_df["Accuracy"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy on held-out test set")
    ax.set_title("Model Accuracy Comparison")
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(out / "model_accuracy_comparison.png", dpi=180)
    plt.close(fig)

    # Confusion matrix
    labels = ["Movie", "TV Show"]
    matrix = confusion_matrix(y_test, best_predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, interpolation="nearest")
    ax.set_title(f"Confusion Matrix – {best_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / "best_model_confusion_matrix.png", dpi=180)
    plt.close(fig)

    # Class-wise precision / recall / F1
    report = classification_report(y_test, best_predictions, labels=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T.loc[labels, ["precision", "recall", "f1-score"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    report_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"Class-wise Evaluation – {best_name}")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "classwise_metrics.png", dpi=180)
    plt.close(fig)

    # Contact sheet / summary evidence page
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].bar(counts.index, counts.values)
    axes[0, 0].set_title("Dataset target distribution")
    axes[0, 0].set_ylabel("Titles")
    accuracy_df = results.sort_values("Accuracy", ascending=True)
    axes[0, 1].barh(accuracy_df["Model"], accuracy_df["Accuracy"])
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_title("Test accuracy by model")
    for bar in axes[0, 1].containers:
        axes[0, 1].bar_label(bar, fmt="%.3f", padding=2)
    axes[1, 0].imshow(matrix, interpolation="nearest")
    axes[1, 0].set_title("Best model confusion matrix")
    axes[1, 0].set_xticks(range(2), labels=labels)
    axes[1, 0].set_yticks(range(2), labels=labels)
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, f"{matrix[i, j]:,}", ha="center", va="center")
    axes[1, 1].axis("off")
    best = results[results["Model"] == best_name].iloc[0]
    lines = [
        "TASK 2 – CONTENT TYPE PREDICTION",
        f"Best model: {best_name}",
        f"Accuracy: {best['Accuracy']:.3f}",
        f"Macro precision: {best['Precision (macro)']:.3f}",
        f"Macro recall: {best['Recall (macro)']:.3f}",
        f"Macro F1: {best['F1 (macro)']:.3f}",
        "",
        "Target: Movie vs TV Show",
        "Validation: stratified 80/20 holdout",
        "Duration, title and show_id excluded to avoid leakage.",
    ]
    axes[1, 1].text(0.02, 0.95, "\n".join(lines), va="top", fontsize=11, family="monospace")
    fig.suptitle("Auspify ML Internship – Task 2 Results", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "Task_2_Results_Snapshot.png", dpi=180)
    plt.close(fig)


def run_project(dataset_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    screenshots_dir = output / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    outputs_dir = output / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    X, y, df = prepare_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    results, fitted, predictions = evaluate_models(X_train, X_test, y_train, y_test)
    best_name = results.iloc[0]["Model"]
    best_pipeline = fitted[best_name]
    best_predictions = predictions[best_name]

    results.to_csv(outputs_dir / "model_comparison.csv", index=False)
    report = classification_report(y_test, best_predictions, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(outputs_dir / "classification_report.csv")
    cm = confusion_matrix(y_test, best_predictions, labels=["Movie", "TV Show"])
    pd.DataFrame(cm, index=["Actual Movie", "Actual TV Show"], columns=["Predicted Movie", "Predicted TV Show"]).to_csv(
        outputs_dir / "confusion_matrix.csv"
    )
    prediction_table = X_test.copy()
    prediction_table["actual_type"] = y_test
    prediction_table["predicted_type"] = best_predictions
    prediction_table["correct"] = prediction_table["actual_type"] == prediction_table["predicted_type"]
    prediction_table.head(500).to_csv(outputs_dir / "sample_test_predictions.csv", index=False)
    df[TARGET].value_counts().rename_axis("content_type").reset_index(name="count").to_csv(
        outputs_dir / "content_type_counts.csv", index=False
    )
    joblib.dump(best_pipeline, outputs_dir / "best_content_type_model.joblib")

    create_visuals(df, results, best_name, y_test, best_predictions, screenshots_dir)

    # Robustness check: re-fit Logistic Regression on the same split without genre labels.
    # This reveals how strongly TV-specific genre tags in `listed_in` help the prediction.
    X_without_genres = X.drop(columns=["listed_in"])
    ablation_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(include_genres=False)),
            ("classifier", LogisticRegression(
                max_iter=2000, class_weight="balanced", C=1.0,
                solver="liblinear", random_state=RANDOM_STATE
            )),
        ]
    )
    ablation_model.fit(X_without_genres.loc[X_train.index], y_train)
    ablation_pred = ablation_model.predict(X_without_genres.loc[X_test.index])
    ablation_result = {
        "Feature set": "All selected metadata",
        "Accuracy": float(accuracy_score(y_test, best_predictions)),
        "Macro F1": float(f1_score(y_test, best_predictions, average="macro", zero_division=0)),
    }
    no_genre_result = {
        "Feature set": "Without listed_in genres",
        "Accuracy": float(accuracy_score(y_test, ablation_pred)),
        "Macro F1": float(f1_score(y_test, ablation_pred, average="macro", zero_division=0)),
    }
    ablation_df = pd.DataFrame([ablation_result, no_genre_result])
    ablation_df.to_csv(outputs_dir / "genre_ablation_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(ablation_df["Feature set"], ablation_df["Accuracy"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy on the same held-out test set")
    ax.set_title("Feature Ablation: Effect of Genre Metadata")
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(screenshots_dir / "genre_feature_ablation.png", dpi=180)
    plt.close(fig)

    # Dataset preview evidence image (sample rows, not a fabricated notebook screenshot).
    preview = df[["show_id", "type", "title", "release_year", "rating", "listed_in"]].head(6)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis("off")
    table = ax.table(cellText=preview.values, colLabels=preview.columns, cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.7)
    ax.set_title("Netflix Dataset – Sample Records", pad=18)
    fig.tight_layout()
    fig.savefig(screenshots_dir / "dataset_preview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "rows_used": int(len(df)),
        "features_used": FEATURE_COLUMNS,
        "target": TARGET,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_model": best_name,
        "metrics": {k: float(v) for k, v in results.iloc[0].drop(labels=["Model"]).to_dict().items()},
        "accuracy_without_listed_in_genres": float(no_genre_result["Accuracy"]),
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
    }
    (outputs_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"data": df, "X": X, "y": y, "X_train": X_train, "X_test": X_test, "y_train": y_train,
            "y_test": y_test, "results": results, "best_name": best_name,
            "best_pipeline": best_pipeline, "best_predictions": best_predictions,
            "ablation_results": ablation_df, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Netflix content type classifiers.")
    parser.add_argument("--dataset", default="Dataset.csv", help="Path to the Netflix Dataset.csv file")
    parser.add_argument("--output", default=".", help="Folder in which outputs and screenshots will be saved")
    args = parser.parse_args()
    result = run_project(args.dataset, args.output)
    print("\nModel comparison (held-out test set):")
    print(result["results"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nBest model: {result['best_name']}")
    print(f"Files saved under: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

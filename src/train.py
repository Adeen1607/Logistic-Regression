"""Train and evaluate a leakage-safe credit-card fraud baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "Class"
SCALED_FEATURES = ["Time", "Amount"]
ANONYMIZED_FEATURES = [f"V{number}" for number in range(1, 29)]
EXPECTED_COLUMNS = SCALED_FEATURES + ANONYMIZED_FEATURES + [TARGET]


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path)
    missing = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    frame = frame[EXPECTED_COLUMNS].copy()
    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values; investigate before training.")
    if not frame[TARGET].isin([0, 1]).all():
        raise ValueError("Class must contain only zero and one.")
    if frame[TARGET].nunique() != 2:
        raise ValueError("Both legitimate and fraud observations are required.")

    return frame.drop(columns=TARGET), frame[TARGET].astype(int)


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features,
        target,
        test_size=0.40,
        stratify=target,
        random_state=seed,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=0.50,
        stratify=y_holdout,
        random_state=seed,
    )
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def build_pipeline(seed: int) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ("scaled", StandardScaler(), SCALED_FEATURES),
            ("anonymized", "passthrough", ANONYMIZED_FEATURES),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        random_state=seed,
        solver="lbfgs",
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


def choose_threshold(
    labels: pd.Series,
    probabilities: pd.Series,
    minimum_recall: float,
) -> tuple[float, float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    candidates = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        }
    )
    eligible = candidates[candidates["recall"] >= minimum_recall]
    if eligible.empty:
        raise ValueError(
            f"No threshold satisfies minimum recall of {minimum_recall:.2f}."
        )

    selected = eligible.sort_values(
        ["precision", "threshold"], ascending=[False, False]
    ).iloc[0]
    return (
        float(selected["threshold"]),
        float(selected["precision"]),
        float(selected["recall"]),
    )


def calculate_metrics(
    labels: pd.Series,
    probabilities: pd.Series,
    predictions: pd.Series,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def save_curves(
    labels: pd.Series,
    probabilities: pd.Series,
    threshold: float,
    output_dir: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recall, precision, color="#2563EB")
    axis.set(
        title=f"Precision–Recall Curve (threshold={threshold:.4f})",
        xlabel="Recall",
        ylabel="Precision",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close(figure)

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(false_positive_rate, true_positive_rate, color="#059669")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#64748B")
    axis.set(
        title="Receiver Operating Characteristic",
        xlabel="False-positive rate",
        ylabel="True-positive rate",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(figure)


def save_confusion_matrix(
    labels: pd.Series,
    predictions: pd.Series,
    output_dir: Path,
) -> None:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=axis,
    )
    axis.set(title="Held-out Test Confusion Matrix", xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def train(data_path: Path, output_dir: Path, minimum_recall: float, seed: int) -> None:
    if not 0 < minimum_recall <= 1:
        raise ValueError("--minimum-recall must be greater than zero and at most one.")

    output_dir.mkdir(parents=True, exist_ok=True)
    features, target = load_data(data_path)
    x_train, x_validation, x_test, y_train, y_validation, y_test = split_data(
        features, target, seed
    )

    pipeline = build_pipeline(seed)
    pipeline.fit(x_train, y_train)

    validation_probability = pipeline.predict_proba(x_validation)[:, 1]
    threshold, validation_precision, validation_recall = choose_threshold(
        y_validation,
        pd.Series(validation_probability, index=y_validation.index),
        minimum_recall,
    )

    test_probability = pd.Series(
        pipeline.predict_proba(x_test)[:, 1],
        index=y_test.index,
        name="fraud_probability",
    )
    test_prediction = (test_probability >= threshold).astype(int)
    metrics = calculate_metrics(y_test, test_probability, test_prediction)
    metrics.update(
        {
            "decision_threshold": threshold,
            "validation_precision_at_threshold": validation_precision,
            "validation_recall_at_threshold": validation_recall,
            "train_rows": len(x_train),
            "validation_rows": len(x_validation),
            "test_rows": len(x_test),
            "random_seed": seed,
        }
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    predictions = pd.DataFrame(
        {
            "actual": y_test,
            "fraud_probability": test_probability,
            "predicted": test_prediction,
        }
    ).sort_index()
    predictions.to_csv(output_dir / "test_predictions.csv", index_label="row_index")

    joblib.dump(pipeline, output_dir / "fraud_logistic_pipeline.joblib")
    save_curves(y_test, test_probability, threshold, output_dir)
    save_confusion_matrix(y_test, test_prediction, output_dir)
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an imbalanced credit-card fraud classifier."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--minimum-recall", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(
        arguments.data,
        arguments.output,
        arguments.minimum_recall,
        arguments.seed,
    )

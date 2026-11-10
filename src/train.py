"""Train, evaluate, and plot the classification models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .logistic_regression_numpy import LogisticRegressionNumpy


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    # sklearn encodes malignant as 0 and benign as 1; use 1 for malignant.
    target = (dataset.target == 0).astype(int)
    return features, target.rename("malignant")


def build_models(random_state: int) -> dict[str, object]:
    scaled = lambda estimator: Pipeline(
        [("scaler", StandardScaler()), ("model", estimator)]
    )
    return {
        "Logistic Regression": scaled(LogisticRegression(max_iter=2_000, random_state=random_state)),
        "KNN": scaled(KNeighborsClassifier(n_neighbors=5)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "SVM": scaled(SVC(kernel="rbf", random_state=random_state)),
        "Logistic Regression (NumPy)": scaled(LogisticRegressionNumpy()),
    }


def evaluate_models(
    models: dict[str, object],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, float | str]] = []
    matrices: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="binary", zero_division=0
        )
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )
        matrices[name] = confusion_matrix(y_test, predictions)
    return pd.DataFrame(rows).sort_values("f1_score", ascending=False), matrices


def save_plots(results: pd.DataFrame, matrices: dict[str, np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = results.set_index("model")[["accuracy", "precision", "recall", "f1_score"]]
    ax = metrics.plot.bar(figsize=(13, 6), ylim=(0.7, 1.02), title="Model comparison")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=150)
    plt.close()

    columns = 3
    rows = int(np.ceil(len(matrices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows))
    for axis, (name, matrix) in zip(np.asarray(axes).reshape(-1), matrices.items()):
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axis,
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
        )
        axis.set_title(name)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
    for axis in np.asarray(axes).reshape(-1)[len(matrices):]:
        axis.remove()
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=150)
    plt.close()


def run(random_state: int = 42, output_dir: Path = Path("results")) -> pd.DataFrame:
    features, target = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=random_state
    )
    results, matrices = evaluate_models(
        build_models(random_state), x_train, x_test, y_train, y_test
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "metrics.csv", index=False)
    save_plots(results, matrices, output_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    print(run(args.random_state, args.output_dir).to_string(index=False))


if __name__ == "__main__":
    main()

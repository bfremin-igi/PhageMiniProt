"""Train an XGBoost classifier on positive/negative embedding matrices."""

import pathlib

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

RANDOM_STATE = 42

# Match Methods defaults (200 trees, lr 0.05, depth 8, 10-fold CV).
DEFAULT_N_ESTIMATORS = 200
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_MAX_DEPTH = 8
DEFAULT_N_SPLITS = 10
DEFAULT_HOLDOUT_FRACTION = 0.15


def calc_metrics(y_true, y_pred, y_pred_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }


def print_metrics(metrics, header=None):
    output = "\n".join(f"{name.capitalize()}: {value:.4f}" for name, value in metrics.items())
    if header:
        output = f"{header}\n{output}"
    print(output)


def print_classification_report(y_true, y_pred):
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


class EmbeddingsClassifier:
    def __init__(self, model=None, verbose=False):
        self.model = model
        self.verbose = verbose

    @classmethod
    def create(
        cls,
        n_estimators=DEFAULT_N_ESTIMATORS,
        learning_rate=DEFAULT_LEARNING_RATE,
        max_depth=DEFAULT_MAX_DEPTH,
        verbose=False,
        **kwargs,
    ):
        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            **kwargs,
        )
        return cls(model, verbose=verbose)

    def train_with_cv(self, X, y, n_splits=DEFAULT_N_SPLITS):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        fold_metrics = {
            "train_accuracy": [],
            "train_precision": [],
            "train_recall": [],
            "train_f1_score": [],
            "train_roc_auc": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1_score": [],
            "val_roc_auc": [],
        }

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            self.model.fit(X_train, y_train)

            y_train_pred = self.model.predict(X_train)
            y_train_pred_proba = self.model.predict_proba(X_train)[:, 1]
            y_val_pred = self.model.predict(X_val)
            y_val_pred_proba = self.model.predict_proba(X_val)[:, 1]

            train_metrics = calc_metrics(y_train, y_train_pred, y_train_pred_proba)
            val_metrics = calc_metrics(y_val, y_val_pred, y_val_pred_proba)

            for metric in fold_metrics:
                key = metric.replace("train_", "").replace("val_", "")
                if metric.startswith("train_"):
                    fold_metrics[metric].append(train_metrics[key])
                else:
                    fold_metrics[metric].append(val_metrics[key])

            if self.verbose:
                print_metrics(train_metrics, header=f"Fold {fold + 1} - Training Metrics")
                print_metrics(val_metrics, header=f"Fold {fold + 1} - Validation Metrics")

        avg_train = {
            m: np.mean(v) for m, v in fold_metrics.items() if m.startswith("train_")
        }
        avg_val = {m: np.mean(v) for m, v in fold_metrics.items() if m.startswith("val_")}
        return avg_train, avg_val

    def save(self, model_dirpath):
        model_dirpath = pathlib.Path(model_dirpath)
        model_dirpath.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_dirpath / "model.joblib")

    @classmethod
    def load(cls, model_dirpath):
        model = joblib.load(pathlib.Path(model_dirpath) / "model.joblib")
        return cls(model)


def train_with_holdout(
    X,
    y,
    holdout_fraction=DEFAULT_HOLDOUT_FRACTION,
    n_estimators=DEFAULT_N_ESTIMATORS,
    learning_rate=DEFAULT_LEARNING_RATE,
    max_depth=DEFAULT_MAX_DEPTH,
    n_splits=DEFAULT_N_SPLITS,
    verbose=True,
):
    """
    Stratified random holdout + CV on the remainder, then refit and score holdout.

    Note: the published PhageMiniProt numbers used a cluster-aware 90:10 split.
    This CLI path is a convenient retrain utility, not a paper reproduction.
    Bundled models under phageminiprot/models/ are the paper models.
    """
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X, y, test_size=holdout_fraction, stratify=y, random_state=RANDOM_STATE
    )

    classifier = EmbeddingsClassifier.create(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        verbose=verbose,
    )
    avg_train_metrics, avg_val_metrics = classifier.train_with_cv(
        X_train_full, y_train_full, n_splits=n_splits
    )

    classifier.model.fit(X_train_full, y_train_full)
    y_holdout_pred = classifier.model.predict(X_holdout)
    y_holdout_pred_proba = classifier.model.predict_proba(X_holdout)[:, 1]
    holdout_metrics = calc_metrics(y_holdout, y_holdout_pred, y_holdout_pred_proba)

    if verbose:
        print_classification_report(y_holdout, y_holdout_pred)

    return classifier, avg_train_metrics, avg_val_metrics, holdout_metrics


def main(
    positive_class_filepath,
    negative_class_filepath,
    model_dirpath,
    n_estimators=DEFAULT_N_ESTIMATORS,
    learning_rate=DEFAULT_LEARNING_RATE,
    max_depth=DEFAULT_MAX_DEPTH,
    n_splits=DEFAULT_N_SPLITS,
    holdout_fraction=DEFAULT_HOLDOUT_FRACTION,
):
    positive_embeddings = np.load(positive_class_filepath)
    negative_embeddings = np.load(negative_class_filepath)

    X = np.vstack([positive_embeddings, negative_embeddings])
    y = np.hstack(
        [np.ones(len(positive_embeddings)), np.zeros(len(negative_embeddings))]
    )

    classifier, avg_train_metrics, avg_val_metrics, holdout_metrics = train_with_holdout(
        X,
        y,
        holdout_fraction=holdout_fraction,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_splits=n_splits,
        verbose=True,
    )
    classifier.save(model_dirpath)

    print_metrics(avg_train_metrics, header="Average Training Metrics")
    print_metrics(avg_val_metrics, header="Average Validation Metrics")
    print_metrics(holdout_metrics, header="Independent Holdout Metrics")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train an XGBoost classifier on embeddings."
    )
    parser.add_argument("--positive-class-filepath", required=True)
    parser.add_argument("--negative-class-filepath", required=True)
    parser.add_argument("--model-dirpath", required=True)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument(
        "--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION
    )
    args = parser.parse_args()
    main(**vars(args))

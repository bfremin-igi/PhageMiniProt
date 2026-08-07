import pathlib
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier


RANDOM_STATE = 42

# NOTE: these now match the Methods section (200 estimators, lr 0.05, depth 8,
# 10-fold CV). Previously this module (and the CLI, before `train` was wired
# up at all) defaulted to vanilla XGBoost/argparse values (100/0.1/6/5-fold),
# which do not reproduce the numbers reported in the paper.
DEFAULT_N_ESTIMATORS = 200
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_MAX_DEPTH = 8
DEFAULT_N_SPLITS = 10
DEFAULT_HOLDOUT_FRACTION = 0.15  # CONFIRM against whatever was actually used for the paper


def calc_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate a variety of performance metrics given true labels and predicted values.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }
    return metrics


def print_metrics(metrics, header=None):
    """
    Print metrics in a readable format.
    """
    output = "\n".join([f"{metric.capitalize()}: {value:.4f}" for metric, value in metrics.items()])
    if header:
        output = f"{header}\n{output}"
    print(output)


def print_classification_report(y_true, y_pred):
    """
    Print a detailed classification report including confusion matrix.
    """
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


class EmbeddingsClassifier:
    """
    A classifier for embeddings using XGBoost.
    """

    def __init__(self, model=None, verbose=False):
        self.model = model
        self.verbose = verbose

    @classmethod
    def create(cls, n_estimators=DEFAULT_N_ESTIMATORS, learning_rate=DEFAULT_LEARNING_RATE,
               max_depth=DEFAULT_MAX_DEPTH, verbose=False, **kwargs):
        """
        Initialize an XGBoost model with given hyperparameters.
        """
        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            eval_metric="logloss",  # Avoids warning in sklearn interface
            **kwargs,
        )
        return cls(model, verbose=verbose)

    def train_with_cv(self, X, y, n_splits=DEFAULT_N_SPLITS):
        """
        Perform k-fold cross-validation and output the metrics for each fold.
        """
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        fold_metrics = {
            "train_accuracy": [], "train_precision": [], "train_recall": [], "train_f1_score": [], "train_roc_auc": [],
            "val_accuracy": [], "val_precision": [], "val_recall": [], "val_f1_score": [], "val_roc_auc": []
        }

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train the model
            self.model.fit(X_train, y_train)

            # Train predictions
            y_train_pred = self.model.predict(X_train)
            y_train_pred_proba = self.model.predict_proba(X_train)[:, 1]

            # Validation predictions
            y_val_pred = self.model.predict(X_val)
            y_val_pred_proba = self.model.predict_proba(X_val)[:, 1]

            # Calculate metrics for this fold
            train_metrics = calc_metrics(y_train, y_train_pred, y_train_pred_proba)
            val_metrics = calc_metrics(y_val, y_val_pred, y_val_pred_proba)

            # Append the metrics for each fold
            for metric in fold_metrics:
                if "train" in metric:
                    fold_metrics[metric].append(train_metrics[metric.replace("train_", "")])
                else:
                    fold_metrics[metric].append(val_metrics[metric.replace("val_", "")])

            if self.verbose:
                print_metrics(train_metrics, header=f"Fold {fold + 1} - Training Metrics")
                print_metrics(val_metrics, header=f"Fold {fold + 1} - Validation Metrics")

        # Average metrics across all folds
        avg_train_metrics = {metric: np.mean(values) for metric, values in fold_metrics.items() if "train" in metric}
        avg_val_metrics = {metric: np.mean(values) for metric, values in fold_metrics.items() if "val" in metric}

        return avg_train_metrics, avg_val_metrics

    def save(self, model_dirpath):
        """
        Save the trained model to disk.
        """
        model_dirpath = pathlib.Path(model_dirpath)
        model_dirpath.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_dirpath / "model.joblib")

    @classmethod
    def load(cls, model_dirpath):
        """
        Load a previously saved model from disk.
        """
        model_dirpath = pathlib.Path(model_dirpath)
        model = joblib.load(model_dirpath / "model.joblib")
        return cls(model)


def train_with_holdout(X, y, holdout_fraction=DEFAULT_HOLDOUT_FRACTION,
                        n_estimators=DEFAULT_N_ESTIMATORS, learning_rate=DEFAULT_LEARNING_RATE,
                        max_depth=DEFAULT_MAX_DEPTH, n_splits=DEFAULT_N_SPLITS, verbose=True):
    """
    Carve out an independent holdout set, run stratified k-fold CV on the
    remainder, then refit on the full CV set and evaluate once on the
    untouched holdout set.

    IMPORTANT: this uses a naive stratified random split (train_test_split).
    If the paper's holdout set was constructed differently -- e.g. split by
    genome/source to avoid near-duplicate/homologous sequences leaking
    between train and holdout -- swap this out for that logic instead, or
    this will not exactly reproduce the reported holdout number.
    """
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X, y, test_size=holdout_fraction, stratify=y, random_state=RANDOM_STATE
    )

    classifier = EmbeddingsClassifier.create(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, verbose=verbose
    )

    avg_train_metrics, avg_val_metrics = classifier.train_with_cv(
        X_train_full, y_train_full, n_splits=n_splits
    )

    # Refit on the full CV pool (all folds' data), then score once on the holdout set
    classifier.model.fit(X_train_full, y_train_full)
    y_holdout_pred = classifier.model.predict(X_holdout)
    y_holdout_pred_proba = classifier.model.predict_proba(X_holdout)[:, 1]
    holdout_metrics = calc_metrics(y_holdout, y_holdout_pred, y_holdout_pred_proba)

    if verbose:
        print_classification_report(y_holdout, y_holdout_pred)

    return classifier, avg_train_metrics, avg_val_metrics, holdout_metrics


def main(positive_class_filepath, negative_class_filepath, model_dirpath,
         n_estimators=DEFAULT_N_ESTIMATORS, learning_rate=DEFAULT_LEARNING_RATE,
         max_depth=DEFAULT_MAX_DEPTH, n_splits=DEFAULT_N_SPLITS,
         holdout_fraction=DEFAULT_HOLDOUT_FRACTION):
    """
    Entry point used by the CLI's `train` command.
    """
    # Load embeddings
    positive_embeddings = np.load(positive_class_filepath)
    negative_embeddings = np.load(negative_class_filepath)

    # Create labels
    X = np.vstack([positive_embeddings, negative_embeddings])
    y = np.hstack([np.ones(len(positive_embeddings)), np.zeros(len(negative_embeddings))])

    classifier, avg_train_metrics, avg_val_metrics, holdout_metrics = train_with_holdout(
        X, y,
        holdout_fraction=holdout_fraction,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_splits=n_splits,
        verbose=True,
    )

    # Save model (trained on the CV pool, i.e. excluding the holdout set)
    classifier.save(model_dirpath)

    # Print average metrics
    print_metrics(avg_train_metrics, header="Average Training Metrics")
    print_metrics(avg_val_metrics, header="Average Validation Metrics")
    print_metrics(holdout_metrics, header="Independent Holdout Metrics")


# Command-line script functionality (kept for running this file directly,
# e.g. `python train.py ...`, in addition to `PhageMiniProt train ...`)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train an XGBoost classifier on embeddings.")
    parser.add_argument("--positive-class-filepath", required=True, help="Path to positive embeddings file (npy format).")
    parser.add_argument("--negative-class-filepath", required=True, help="Path to negative embeddings file (npy format).")
    parser.add_argument("--model-dirpath", required=True, help="Directory to save the trained model.")
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS, help="Number of trees in XGBoost.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate for XGBoost.")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH, help="Maximum depth of XGBoost trees.")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS, help="Number of folds for cross-validation.")
    parser.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION,
                         help="Fraction of data reserved as an independent holdout set.")
    args = parser.parse_args()

    main(
        positive_class_filepath=args.positive_class_filepath,
        negative_class_filepath=args.negative_class_filepath,
        model_dirpath=args.model_dirpath,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        n_splits=args.n_splits,
        holdout_fraction=args.holdout_fraction,
    )

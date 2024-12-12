import pathlib
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
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
    def create(cls, n_estimators=100, learning_rate=0.1, max_depth=6, **kwargs):
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
        return cls(model)

    def train_with_cv(self, X, y, n_splits=5):
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


# Command-line script functionality
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train an XGBoost classifier on embeddings.")
    parser.add_argument("--positive-class-filepath", required=True, help="Path to positive embeddings file (npy format).")
    parser.add_argument("--negative-class-filepath", required=True, help="Path to negative embeddings file (npy format).")
    parser.add_argument("--model-dirpath", required=True, help="Directory to save the trained model.")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees in XGBoost.")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate for XGBoost.")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum depth of XGBoost trees.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds for cross-validation.")
    args = parser.parse_args()

    # Load embeddings
    positive_embeddings = np.load(args.positive_class_filepath)
    negative_embeddings = np.load(args.negative_class_filepath)

    # Create labels
    X = np.vstack([positive_embeddings, negative_embeddings])
    y = np.hstack([np.ones(len(positive_embeddings)), np.zeros(len(negative_embeddings))])

    # Train model with cross-validation
    classifier = EmbeddingsClassifier.create(
        n_estimators=args.n_estimators, learning_rate=args.learning_rate, max_depth=args.max_depth, verbose=True
    )
    avg_train_metrics, avg_val_metrics = classifier.train_with_cv(X, y, n_splits=args.n_splits)

    # Save model
    classifier.save(args.model_dirpath)

    # Print average metrics
    print_metrics(avg_train_metrics, header="Average Training Metrics")
    print_metrics(avg_val_metrics, header="Average Validation Metrics")


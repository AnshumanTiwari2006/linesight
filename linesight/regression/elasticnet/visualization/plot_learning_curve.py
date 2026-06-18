import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.utils.colors import FIT_LINE, RESIDUAL_POS


def plot_learning_curve(self, X, y, cv_splits: int = 5,
                        train_sizes=None, display: bool = True):
    """
    Show how model performance changes as training dataset size grows.

    What is drawn
    -------------
    - X-axis: number of training samples used
    - Y-axis: R² score (or accuracy for logistic)
    - Blue line: training score at each dataset size
    - Red line: validation score at each dataset size

    What this teaches
    -----------------
    - If training score is high but validation score is low at ALL sizes:
      the model is overfitting regardless of data size
    - If both scores are low and flat: the model is underfitting — more
      data won't help, the model type is wrong
    - If validation score is rising and approaching training score as n grows:
      getting more data WILL help

    Algorithm
    ---------
    For each train_size in train_sizes:
        1. Take first train_size samples as training set
        2. Use the remaining as validation set
        3. Fit a temporary model on training set
        4. Score on both sets
        5. Repeat cv_splits times with different random subsets, take mean

    Parameters
    ----------
    X : array-like
    y : array-like
    cv_splits : int, default 5
        Number of random subsets to average for each train size.
        Higher = smoother curve but slower.
    train_sizes : list of int, optional
        Default: 10 evenly-spaced sizes from 10% to 90% of dataset.
    display : bool, default True
    """
    self._check_fitted("plot_learning_curve")
    X, y = _validate_Xy(X, y)
    n = X.shape[0]

    if train_sizes is None:
        # 10 sizes from 10% to 90% of dataset
        train_sizes = [int(n * frac) for frac in np.linspace(0.1, 0.9, 10)]
        train_sizes = [max(s, 5) for s in train_sizes]  # minimum 5 samples

    train_scores = []
    val_scores = []

    # Reuse the same class as self to create temporary models
    ModelClass = self.__class__
    init_params = {
        'epochs': self.epochs,
    }
    if hasattr(self, 'learning_rate'):
        init_params['learning_rate'] = self.learning_rate
    if hasattr(self, 'degree'):
        init_params['degree'] = self.degree
    if hasattr(self, 'alpha'):
        init_params['alpha'] = self.alpha
    if hasattr(self, 'l1_ratio'):
        init_params['l1_ratio'] = self.l1_ratio
    if hasattr(self, 'normalize'):
        init_params['normalize'] = self.normalize

    from linesight.metrics import r2, accuracy

    for size in train_sizes:
        split_train_scores = []
        split_val_scores = []

        for _ in range(cv_splits):
            perm = np.random.permutation(n)
            train_idx = perm[:size]
            val_idx = perm[size:]

            if len(val_idx) == 0:
                continue

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            tmp = ModelClass(**init_params)
            tmp.fit(X_tr, y_tr)

            y_pred_tr = tmp.predict(X_tr)
            y_pred_val = tmp.predict(X_val)

            # Use accuracy for logistic, r2 for everything else
            if hasattr(self, '_is_classifier') and self._is_classifier:
                split_train_scores.append(accuracy(y_tr, y_pred_tr))
                split_val_scores.append(accuracy(y_val, y_pred_val))
            else:
                split_train_scores.append(r2(y_tr, y_pred_tr))
                split_val_scores.append(r2(y_val, y_pred_val))

        train_scores.append(np.mean(split_train_scores))
        val_scores.append(np.mean(split_val_scores))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_scores, color=FIT_LINE, linewidth=2,
            marker='o', markersize=4, label="Training score")
    ax.plot(train_sizes, val_scores, color=RESIDUAL_POS, linewidth=2,
            marker='o', markersize=4, label="Validation score", linestyle='--')

    ax.set_xlabel("Training set size")
    ax.set_ylabel("R² score" if not (hasattr(self, '_is_classifier') and self._is_classifier)
                  else "Accuracy")
    ax.set_title("Learning curve")
    ax.legend()
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(y=1.0, color="#cccccc", linewidth=0.8, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)

    if display:
        return self.show(fig=fig)
    return fig

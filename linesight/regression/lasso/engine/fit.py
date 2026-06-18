import numpy as np
import warnings
from linesight.utils.validators import _validate_Xy
from linesight.utils.history import TrainingHistory
from linesight.exceptions import LineSightConvergenceWarning, LineSightGradientError
from linesight.regression.lasso.engine.soft_threshold import _soft_threshold
from linesight.regression.lasso.engine.compute_loss import _compute_loss


def _fit(self, X, y):
    """Train Lasso with coordinate descent + soft-thresholding."""
    self._validate_hyperparams()

    feature_names = list(X.columns) if hasattr(X, "columns") else None
    X, y = _validate_Xy(X, y)
    n_samples, n_features = X.shape
    self._n_features_in_ = n_features
    self.feature_names_in_ = feature_names or [f"x{i}" for i in range(n_features)]

    # Feature scaling
    if getattr(self, "normalize", False):
        self._X_mean = X.mean(axis=0)
        self._X_std  = X.std(axis=0)
        self._X_std[self._X_std == 0] = 1.0
        X = (X - self._X_mean) / self._X_std
    else:
        self._X_mean = None
        self._X_std  = None

    self.weights = np.zeros(n_features)
    self.bias = 0.0

    losses, weights, biases = [], [], []

    with np.errstate(over='ignore', invalid='ignore'):
        for epoch in range(self.epochs):
            self.bias = np.mean(y - np.dot(X, self.weights))

            for j in range(n_features):
                w_j = self.weights[j]
                y_excl_j = np.dot(X, self.weights) - X[:, j] * w_j + self.bias
                rho_j = (1 / n_samples) * np.dot(X[:, j], y - y_excl_j)
                x_j_sq = np.mean(X[:, j] ** 2)
                self.weights[j] = 0.0 if x_j_sq == 0 else _soft_threshold(rho_j, self.alpha) / x_j_sq

            loss = _compute_loss(self, X, y)

            if not np.isfinite(loss):
                raise LineSightGradientError(
                    f"Training failed: loss exploded to infinity at epoch {epoch}. "
                    f"Coordinate descent is usually stable, but extremely large feature values can cause overflow. "
                    f"Try scaling your features first."
                )

            if self.store_history:
                losses.append(loss)
                weights.append(self.weights.copy())
                biases.append(self.bias)

    converged = True
    if self.store_history and len(losses) >= 2:
        if losses[-1] < losses[-2]:
            converged = False
            warnings.warn("Model hasn't fully converged. Loss still decreasing.", LineSightConvergenceWarning, stacklevel=2)

    self._history = TrainingHistory(
        losses=losses, weights=weights, biases=biases,
        learning_rate=0.0, epochs_run=self.epochs, converged=converged
    )
    self._is_fitted = True
    return self

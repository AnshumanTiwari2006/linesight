import numpy as np
import warnings
from linesight.utils.validators import _validate_Xy
from linesight.utils.history import TrainingHistory
from linesight.exceptions import LineSightConvergenceWarning, LineSightGradientError
from linesight.regression.multiple.engine.gradient_step import _gradient_step
from linesight.regression.multiple.engine.compute_loss import _compute_loss


def _fit(self, X, y):
    """
    Train multiple linear regression: y_pred = dot(X, weights) + bias.

    If normalize=True (default False), standardizes each feature to
    mean=0 / std=1 before training. Prevents gradient explosion on
    mixed-scale features. Predictions are automatically un-scaled.
    """
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
        self._X_std[self._X_std == 0] = 1.0   # avoid div-by-zero on constant cols
        X = (X - self._X_mean) / self._X_std
    else:
        self._X_mean = None
        self._X_std  = None

    self.weights = np.zeros(n_features)
    self.bias    = 0.0

    losses, weights, biases = [], [], []

    with np.errstate(over='ignore', invalid='ignore'):
        for epoch in range(self.epochs):
            batch_X, batch_y = _get_batch(X, y, self.batch_size)
            _gradient_step(self, batch_X, batch_y)
            loss = _compute_loss(self, X, y)

            if not np.isfinite(loss):
                raise LineSightGradientError(
                    f"Training failed: gradients exploded to infinity at epoch {epoch}. "
                    f"This happens when features are on very different scales or the learning rate is too high. "
                    f"Try drastically reducing 'learning_rate' (e.g., to 0.0001) "
                    f"or initialize your model with 'normalize=True' to fix this automatically."
                )

            if self.store_history:
                losses.append(loss)
                weights.append(self.weights.copy())
                biases.append(self.bias)

    converged = True
    if self.store_history and len(losses) >= 2:
        if losses[-1] < losses[-2]:
            converged = False
            warnings.warn(
                f"Model hasn't fully converged. Loss was still decreasing at epoch {self.epochs}. "
                "Try increasing 'epochs' or using a larger 'learning_rate'. "
                "If features have very different scales, try normalize=True.",
                LineSightConvergenceWarning, stacklevel=2
            )

    self._history = TrainingHistory(
        losses=losses, weights=weights, biases=biases,
        learning_rate=self.learning_rate, epochs_run=self.epochs, converged=converged
    )
    self._is_fitted = True
    return self


def _get_batch(X, y, batch_size):
    """Return a mini-batch or the full dataset if batch_size is None."""
    if batch_size is None:
        return X, y
    n = X.shape[0]
    if batch_size >= n:
        return X, y   # warn-free clamping
    idx = np.random.choice(n, size=batch_size, replace=False)
    return X[idx], y[idx]

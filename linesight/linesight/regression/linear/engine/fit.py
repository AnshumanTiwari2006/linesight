import numpy as np
import warnings
from linesight.utils.validators import _validate_Xy
from linesight.utils.history import TrainingHistory
from linesight.exceptions import LineSightConvergenceWarning, LineSightGradientError
from linesight.regression.linear.engine.gradient_step import _gradient_step
from linesight.regression.linear.engine.compute_loss import _compute_loss


def _fit(self, X, y):
    """Train simple linear regression with gradient descent (m*x + b)."""
    self._validate_hyperparams()

    feature_names = list(X.columns) if hasattr(X, "columns") else None
    X, y = _validate_Xy(X, y)
    self._n_features_in_ = X.shape[1]
    self.feature_names_in_ = feature_names or [f"x{i}" for i in range(self._n_features_in_)]

    # Tutor-style naming: slope and intercept
    self.m = 0.0
    self.b = 0.0

    losses, weights, biases, gradients = [], [], [], []

    with np.errstate(over='ignore', invalid='ignore'):
        for epoch in range(self.epochs):
            grad_m, grad_b = _gradient_step(self, X, y)
            loss = _compute_loss(self, X, y)

            if not np.isfinite(loss):
                raise LineSightGradientError(
                    f"Training failed: gradients exploded to infinity at epoch {epoch}. "
                    f"This usually happens when the learning rate is too high for the scale of your data. "
                    f"Try reducing 'learning_rate' (e.g., from {self.learning_rate} to {self.learning_rate / 10}) "
                    f"or try normalizing your features."
                )

            if self.store_history:
                losses.append(loss)
                weights.append(self.m)   # scalar — m history
                biases.append(self.b)    # scalar — b history
                gradients.append(np.array([grad_m, grad_b]))

    converged = True
    if self.store_history and len(losses) >= 2:
        if losses[-1] < losses[-2]:
            converged = False
            warnings.warn(
                f"Model hasn't fully converged. Loss was still decreasing at epoch {self.epochs}. "
                "Try increasing 'epochs' or using a larger 'learning_rate'.",
                LineSightConvergenceWarning, stacklevel=2
            )

    self._history = TrainingHistory(
        losses=losses, weights=weights, biases=biases, gradients=gradients,
        learning_rate=self.learning_rate, epochs_run=self.epochs, converged=converged,
    )
    self._is_fitted = True
    return self

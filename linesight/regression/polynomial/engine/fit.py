import numpy as np
import warnings
from linesight.utils.validators import _validate_Xy
from linesight.utils.array_utils import _add_bias_column
from linesight.utils.history import TrainingHistory
from linesight.exceptions import LineSightShapeError, LineSightConvergenceWarning, LineSightGradientError
from linesight.regression.polynomial.engine.expand_features import _expand_features


def _fit(self, X, y):
    """
    Fit polynomial regression by expanding X to degree d then running
    gradient descent on the expanded feature matrix.

    The model learns coefficients [b, a₁, a₂, ..., aᵈ] where:
        ŷ = b + a₁x + a₂x² + ... + aᵈxᵈ

    Stored as self.theta_ shape (degree + 1,):
        theta_[0] = intercept b
        theta_[1] = coefficient for x¹
        theta_[2] = coefficient for x²
        ...

    Parameters
    ----------
    X : array-like, shape (n,) or (n, 1) — single feature only
    y : array-like, shape (n,)

    Raises
    ------
    LineSightShapeError if X has more than 1 feature.
    Polynomial regression in LineSight is intentionally single-feature.
    For multi-feature polynomial, use PolynomialFeatures + MultipleLinearRegression
    (document this clearly).
    """
    self._validate_hyperparams()
    
    if hasattr(X, 'columns') and len(X.columns) > 1:
        raise LineSightShapeError(
            f"PolynomialRegression requires exactly 1 input feature.\n"
            f"Your DataFrame has {len(X.columns)} features.\n"
            f"For multi-feature polynomial: expand features manually, "
            f"then use MultipleLinearRegression."
        )

    X, y = _validate_Xy(X, y)

    if X.shape[1] != 1:
        raise LineSightShapeError(
            f"PolynomialRegression requires exactly 1 input feature.\n"
            f"Your X has {X.shape[1]} features."
        )

    self._n_features_in_ = 1
    self._X_original = X.copy()  # store for visualization axis limits

    X_poly = _expand_features(X, self.degree)   # shape (n, degree)
    
    if getattr(self, "normalize", False):
        self._X_mean = X_poly.mean(axis=0)
        self._X_std = X_poly.std(axis=0)
        self._X_std[self._X_std == 0] = 1.0
        X_poly = (X_poly - self._X_mean) / self._X_std
    else:
        self._X_mean = None
        self._X_std = None

    X_bias = _add_bias_column(X_poly)            # shape (n, degree+1)

    self.theta_ = np.zeros(self.degree + 1)

    losses = []
    coefficients = []

    with np.errstate(over='ignore', invalid='ignore'):
        for epoch in range(self.epochs):
            n = X_bias.shape[0]
            y_pred = X_bias @ self.theta_
            residuals = y_pred - y

            grad = (2 / n) * X_bias.T @ residuals
            self.theta_ -= self.learning_rate * grad

            loss = float(np.mean(residuals ** 2))
            
            if not np.isfinite(loss):
                raise LineSightGradientError(
                    f"Training failed: gradients exploded to infinity at epoch {epoch}. "
                    f"Because polynomial features expand exponentially (e.g. x^{self.degree}), "
                    f"they require an extremely small learning rate if the data is not scaled. "
                    f"Try drastically reducing your learning_rate (e.g., to 0.00001 or smaller) "
                    f"or standardize your X values before fitting."
                )

            if self.store_history:
                losses.append(loss)
                coefficients.append(self.theta_.copy())

    converged = True
    if self.store_history and len(losses) >= 2 and losses[-1] < losses[-2]:
        converged = False
        warnings.warn(
            f"Model hasn't fully converged. Loss was still decreasing at epoch {self.epochs}. "
            f"Try increasing 'epochs' to allow the model to finish learning.",
            LineSightConvergenceWarning, stacklevel=2
        )

    weights = [c[1:] for c in coefficients] if coefficients else []
    biases = [c[0] for c in coefficients] if coefficients else []

    self._history = TrainingHistory(
        losses=losses, weights=weights, biases=biases, gradients=[],
        learning_rate=self.learning_rate, epochs_run=self.epochs,
        converged=converged,
    )
    self._is_fitted = True
    return self

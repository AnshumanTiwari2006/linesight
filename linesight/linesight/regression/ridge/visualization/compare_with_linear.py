import numpy as np
import matplotlib.pyplot as plt
import warnings
from linesight.utils.validators import _validate_Xy


def compare_with_linear(self, X, y, display: bool = True):
    """
    Side-by-side comparison: Ridge (current model) vs unregularized Linear.
    Shows coefficient magnitudes and R^2 for both.
    """
    self._check_fitted("compare_with_linear")
    X, y = _validate_Xy(X, y)

    from linesight.regression.multiple.core import MultipleLinearRegression
    ols = MultipleLinearRegression(
        learning_rate=self.learning_rate, epochs=self.epochs
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols.fit(X, y)

    ridge_coefs = self.weights
    ols_coefs = ols.weights
    names = self.feature_names_in_
    x_pos = np.arange(len(names))

    ridge_score = self.score(X, y)["r2"]
    ols_score = ols.score(X, y)["r2"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x_pos - 0.2, np.abs(ols_coefs), 0.4, color="#888888", label=f"OLS (R^2={ols_score:.4f})")
    ax1.bar(x_pos + 0.2, np.abs(ridge_coefs), 0.4, color="#1a6fcc", label=f"Ridge alpha={self.alpha} (R^2={ridge_score:.4f})")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=30)
    ax1.set_ylabel("|Coefficient|")
    ax1.set_title("Coefficient magnitudes")
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)

    # Actual vs predicted for both
    y_ols = ols.predict(X)
    y_ridge = self.predict(X)
    ax2.scatter(y, y_ols, color="#888888", s=30, alpha=0.6, label=f"OLS")
    ax2.scatter(y, y_ridge, color="#1a6fcc", s=30, alpha=0.6, label=f"Ridge")
    lims = [y.min(), y.max()]
    ax2.plot(lims, lims, 'k--', linewidth=1)
    ax2.set_xlabel("Actual y")
    ax2.set_ylabel("Predicted ŷ")
    ax2.set_title("Actual vs Predicted")
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if display:
        return self.show(fig=fig)
    return fig

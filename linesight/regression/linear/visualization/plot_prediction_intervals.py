import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.utils.colors import DATA_POINTS, FIT_LINE
from linesight.exceptions import LineSightShapeError, LineSightDataWarning
import warnings
from linesight.utils.viz_context import print_viz_context


def plot_prediction_intervals(self, X, y,
                               confidence: float = 0.95,
                               display: bool = True):
    """
    Plot the regression line with confidence and prediction interval bands.

    Two interval types are shown:
    ---------------------------------------------------------------
    Confidence interval (inner, darker band):
        Uncertainty about WHERE THE MEAN LINE IS.
        Formula: ŷ ± t * SE_mean
        SE_mean = s * sqrt(1/n + (x - x̄)² / Σ(xᵢ - x̄)²)
        Narrowest at x=x̄ (the mean of X), widens toward extremes.

    Prediction interval (outer, lighter band):
        Uncertainty about WHERE A NEW INDIVIDUAL POINT WILL FALL.
        Formula: ŷ ± t * SE_pred
        SE_pred = s * sqrt(1 + 1/n + (x - x̄)² / Σ(xᵢ - x̄)²)
        Always wider than confidence interval by the extra "1 +" term.
        This extra term accounts for individual random variation.

    Where:
        s = residual standard error = sqrt(MSE * n / (n-2))
        t = t-distribution critical value for (n-2) degrees of freedom
        n = number of training samples
        x̄ = mean of training X

    The t-distribution is used instead of z=1.96 for finite samples.
    For n > 30 they are nearly identical. For n < 30 the difference matters.

    Parameters
    ----------
    X : array-like, shape (n,) or (n, 1)
    y : array-like, shape (n,)
    confidence : float, default 0.95
        Confidence level. 0.95 = 95% confidence intervals.
        Must be between 0.5 and 0.999.
    display : bool, default True

    Raises
    ------
    LineSightShapeError
        If X has more than 1 feature.
        Message: "plot_prediction_intervals() only works for single-feature models.
        Your X has {n} features.
        For multi-feature models, use plot_actual_vs_predicted() to assess fit quality."

    LineSightShapeError
        If confidence is outside [0.5, 0.999].
        Message: "confidence must be between 0.5 and 0.999.
        Received: {confidence}
        Common values: 0.90 (90%), 0.95 (95%), 0.99 (99%)."

    Warns
    -----
    LineSightDataWarning if n < 10:
        "Only {n} samples. Prediction intervals will be very wide and unreliable.
        Intervals require at least 20-30 samples to be meaningful."
    """
    self._check_fitted("plot_prediction_intervals")
    X, y = _validate_Xy(X, y)

    if X.shape[1] != 1:
        raise LineSightShapeError(
            f"plot_prediction_intervals() only works for single-feature models.\n"
            f"Your X has {X.shape[1]} features.\n"
            f"For multi-feature models, use plot_actual_vs_predicted() instead."
        )

    if not (0.5 <= confidence <= 0.999):
        raise LineSightShapeError(
            f"confidence must be between 0.5 and 0.999.\n"
            f"Received: {confidence}\n"
            f"Common values: 0.90, 0.95, 0.99"
        )

    n = X.shape[0]
    if n < 10:
        warnings.warn(
            f"Only {n} samples. Prediction intervals will be very wide and unreliable.\n"
            f"Intervals require at least 20–30 samples to be meaningful.",
            LineSightDataWarning, stacklevel=2
        )

    x = X.ravel()
    y_pred_train = self.predict(X)

    # Residual standard error
    mse_val = float(np.mean((y - y_pred_train) ** 2))
    s = np.sqrt(mse_val * n / max(n - 2, 1))  # unbiased RSE

    # t critical value — approximate using scipy if available, else use 1.96
    try:
        from scipy import stats
        t_crit = float(stats.t.ppf((1 + confidence) / 2, df=n - 2))
    except ImportError:
        # Fallback: rough approximation for large n
        t_crit = 1.96 if confidence >= 0.94 else (1.645 if confidence >= 0.89 else 1.28)
        warnings.warn(
            "scipy not installed. Using approximate t critical value. "
            "Install scipy for exact intervals: pip install scipy",
            LineSightDataWarning, stacklevel=2
        )

    x_bar = np.mean(x)
    ssx = np.sum((x - x_bar) ** 2)  # sum of squares of x deviations

    # Dense x range for smooth bands
    x_plot = np.linspace(x.min(), x.max(), 300)
    y_plot = self.coef_ * x_plot + self.intercept_

    # Standard error at each x_plot point
    se_mean = s * np.sqrt(1/n + (x_plot - x_bar)**2 / max(ssx, 1e-10))
    se_pred = s * np.sqrt(1 + 1/n + (x_plot - x_bar)**2 / max(ssx, 1e-10))

    ci_lower = y_plot - t_crit * se_mean
    ci_upper = y_plot + t_crit * se_mean
    pi_lower = y_plot - t_crit * se_pred
    pi_upper = y_plot + t_crit * se_pred

    pct = int(confidence * 100)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Prediction interval (outer, lighter)
    ax.fill_between(x_plot, pi_lower, pi_upper,
                    color=FIT_LINE, alpha=0.10,
                    label=f"{pct}% prediction interval\n(where new individual points will fall)")

    # Confidence interval (inner, darker)
    ax.fill_between(x_plot, ci_lower, ci_upper,
                    color=FIT_LINE, alpha=0.25,
                    label=f"{pct}% confidence interval\n(uncertainty about the mean line)")

    # Regression line
    ax.plot(x_plot, y_plot, color=FIT_LINE, linewidth=2, label="Fit")

    # Data points
    ax.scatter(x, y, color=DATA_POINTS, edgecolors="white",
               linewidths=0.8, s=55, zorder=4)

    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title(
        f"{pct}% Confidence & Prediction Intervals\n"
        f"t-critical = {round(t_crit, 3)}  |  RSE = {round(s, 4)}  |  n = {n}"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    print_viz_context(
        diagram="Confidence & Prediction Interval Bands",
        solves="Quantifies uncertainty: how sure are we about the line's position, and where will future points land?",
        theory=(
            "Two distinct bands are shown. The CONFIDENCE INTERVAL is uncertainty about"
            " the TRUE MEAN LINE — it narrows at the center (x=x_bar) because the mean"
            " is estimated most precisely there. The PREDICTION INTERVAL includes additional"
            " uncertainty for individual future observations and is always wider."
            " Both use the t-distribution which corrects for small sample sizes."
        ),
        formula=(
            "CI: y_hat +/- t * s * sqrt(1/n + (x - x_bar)^2 / SSx)\n"
            "PI: y_hat +/- t * s * sqrt(1 + 1/n + (x - x_bar)^2 / SSx)\n"
            "where s = sqrt(MSE * n/(n-2)),  SSx = sum((x_i - x_bar)^2)"
        ),
        constraints=(
            "- X must have exactly 1 feature\n"
            "- Assumes residuals are normally distributed\n"
            "- Assumes homoscedasticity (equal variance across all x)\n"
            "- Intervals widen toward extremes; extrapolation is unreliable"
        ),
        reading=(
            "- Darker inner band = confidence interval (mean line uncertainty).\n"
            "- Lighter outer band = prediction interval (where a NEW point may land).\n"
            "- Bands narrow at x = mean of X (most data, most certainty).\n"
            "- Points outside the outer band are statistical outliers."
        ),
    )
    if display:
        return self.show(fig=fig)
    return fig

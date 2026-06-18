import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.utils.colors import FIT_LINE, RESIDUAL_POS
from linesight.exceptions import LineSightDataWarning
import warnings


def plot_calibration_curve(self, X, y,
                            n_bins: int = 10,
                            display: bool = True):
    """
    Plot a reliability diagram: predicted probability vs actual positive rate.

    Algorithm
    ---------
    1. Get predicted probabilities from predict_proba(X)
    2. Bin samples into n_bins equal-width bins by predicted probability
       (bin 1: p=0.0–0.1, bin 2: p=0.1–0.2, ..., bin 10: p=0.9–1.0)
    3. For each bin: compute mean predicted probability and mean actual y
    4. Plot mean predicted (x-axis) vs mean actual (y-axis)

    Perfect calibration: all points on the diagonal (y=x)
    Overconfident model: points below the diagonal
    Underconfident model: points above the diagonal

    What is drawn
    -------------
    - Diagonal dashed line (perfect calibration)
    - Blue line + dots: actual calibration curve
    - Bar chart at the bottom: sample count per bin
      (thin bins = fewer samples = less reliable estimate)

    Parameters
    ----------
    X : array-like
    y : array-like
    n_bins : int, default 10
        Number of probability bins.
    display : bool, default True

    Raises
    ------
    LineSightShapeError
        If n_bins > 20 or n_bins < 3:
        "n_bins must be between 3 and 20. Received {n}.
        Too few bins loses resolution. Too many bins has sparse counts."

    Warns
    -----
    LineSightDataWarning if any bin has fewer than 5 samples:
        "Bin [{low}, {high}] has only {count} samples. Calibration
        estimate for this bin is unreliable."
    """
    from linesight.exceptions import LineSightShapeError

    self._check_fitted("plot_calibration_curve")

    if not (3 <= n_bins <= 20):
        raise LineSightShapeError(
            f"n_bins must be between 3 and 20. Received {n_bins}.\n"
            f"Too few bins loses resolution. Too many bins creates sparse counts."
        )

    X, y = _validate_Xy(X, y)
    proba = self.predict_proba(X)
    y_int = y.astype(int)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred = []
    mean_actual = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (proba >= lo) & (proba < hi)
        count = int(mask.sum())
        bin_counts.append(count)

        if count == 0:
            mean_pred.append((lo + hi) / 2)
            mean_actual.append(float('nan'))
        else:
            if count < 5:
                warnings.warn(
                    f"Bin [{round(lo, 2)}, {round(hi, 2)}] has only {count} samples.\n"
                    f"Calibration estimate for this bin is unreliable.",
                    LineSightDataWarning, stacklevel=2
                )
            mean_pred.append(float(proba[mask].mean()))
            mean_actual.append(float(y_int[mask].mean()))

    mean_pred = np.array(mean_pred)
    mean_actual = np.array(mean_actual)
    valid = ~np.isnan(mean_actual)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 8),
        gridspec_kw={'height_ratios': [3, 1]}, sharex=True
    )

    # Perfect calibration diagonal
    ax_top.plot([0, 1], [0, 1], color="#aaaaaa", linewidth=1.5,
                linestyle="--", label="Perfect calibration")

    # Actual calibration curve
    ax_top.plot(mean_pred[valid], mean_actual[valid], color=FIT_LINE,
                linewidth=2, marker='o', markersize=7, label="Model calibration")

    # Shade the over/under-confident regions
    ax_top.fill_between([0, 1], [0, 1], [0, 0],
                        alpha=0.04, color="#e05252", label="Overconfident region")
    ax_top.fill_between([0, 1], [0, 1], [1, 1],
                        alpha=0.04, color="#5280e0", label="Underconfident region")

    ax_top.set_ylabel("Fraction of actual positives")
    ax_top.set_title("Calibration curve (reliability diagram)\n"
                     "Points on diagonal = perfectly calibrated")
    ax_top.set_xlim(-0.02, 1.02)
    ax_top.set_ylim(-0.05, 1.05)
    ax_top.legend(fontsize=9)
    ax_top.spines[["top", "right"]].set_visible(False)

    # Bottom: sample counts per bin
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2
                   for i in range(n_bins)]
    ax_bot.bar(bin_centers, bin_counts,
               width=1 / n_bins * 0.85, color=FIT_LINE, alpha=0.6)
    ax_bot.set_xlabel("Mean predicted probability")
    ax_bot.set_ylabel("Samples")
    ax_bot.set_title("Sample count per bin (thin bars = unreliable estimate)")
    ax_bot.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if display:
        return self.show(fig=fig)
    return fig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from linesight.utils.validators import _validate_Xy
from linesight.utils.colors import COLOR_ELASTICNET, COLOR_RIDGE, COLOR_LASSO


def animate_l1_ratio_sweep(self, X, y,
                            n_frames: int = 40,
                            interval: int = 150,
                            display: bool = True):
    """
    Animate coefficient changes as l1_ratio sweeps from 0.0 to 1.0.

    Each frame fits an ElasticNet model with a different l1_ratio and
    shows the resulting coefficient bar chart.

    What is drawn
    -------------
    Bar chart of coefficient magnitudes. Each frame:
    - Title: "l1_ratio = 0.25 — Ridge-dominant" or "l1_ratio = 0.75 — Lasso-dominant"
    - Bar color interpolates from green (Ridge) to red (Lasso)
    - Gray bars = zeroed coefficients
    - Subtitle: count of zeroed features

    Parameters
    ----------
    X : array-like
    y : array-like
    n_frames : int, default 40
        Number of l1_ratio values to animate (evenly spaced 0 to 1).
    interval : int, default 150
    display : bool, default True
    """
    self._check_fitted("animate_l1_ratio_sweep")
    X, y = _validate_Xy(X, y)
    p = X.shape[1]

    feature_names = getattr(self, 'feature_names_in_',
                            [f"x{j}" for j in range(p)])

    from linesight.regression.elasticnet.core import ElasticNetRegression

    l1_ratios = np.linspace(0.0, 1.0, n_frames)

    # Pre-compute all models
    all_coefs = []
    for ratio in l1_ratios:
        tmp = ElasticNetRegression(
            alpha=self.alpha, l1_ratio=float(ratio),
            epochs=self.epochs
        )
        tmp.fit(X, y)
        all_coefs.append(tmp.weights.copy())

    max_abs = max(max(abs(c).max() for c in all_coefs), 0.01) * 1.15

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(feature_names, np.abs(all_coefs[0]),
                  color=COLOR_RIDGE, alpha=0.85)
    title = ax.set_title("")
    zero_text = ax.text(0.98, 0.98, "", transform=ax.transAxes,
                        ha='right', va='top', fontsize=10, color="#e05252")

    ax.set_ylim(0, max_abs)
    ax.set_ylabel("|coefficient|")
    ax.spines[["top", "right"]].set_visible(False)

    def _update(frame):
        ratio = l1_ratios[frame]
        coefs = all_coefs[frame]
        n_zero = int(np.sum(np.abs(coefs) < 1e-6))

        # Color interpolation: green (Ridge) → red (Lasso)
        r = int(42 + (217 - 42) * ratio)
        g = int(157 - (157 - 82) * ratio)
        b = int(78 - (78 - 74) * ratio)
        bar_color = f"#{r:02x}{g:02x}{b:02x}"

        for bar, coef in zip(bars, coefs):
            bar.set_height(abs(coef))
            bar.set_color("#cccccc" if abs(coef) < 1e-6 else bar_color)

        if ratio < 0.3:
            mode = "Ridge-dominant"
        elif ratio > 0.7:
            mode = "Lasso-dominant"
        else:
            mode = "Balanced"

        title.set_text(f"l1_ratio = {round(ratio, 2)} — {mode}  (α={self.alpha})")
        zero_text.set_text(f"{n_zero}/{p} zeroed")
        return bars

    anim = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=interval, blit=False
    )
    plt.tight_layout()

    if display:
        return self.show(animation_obj=anim)
    return anim

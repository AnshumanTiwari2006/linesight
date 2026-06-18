import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.utils.colors import FIT_LINE, RESIDUAL_POS, COLOR_RIDGE


def plot_threshold_sensitivity(self, X, y, display: bool = True):
    """
    Show how precision, recall, F1, and accuracy change as the classification
    threshold shifts from 0 to 1.

    What is drawn
    -------------
    - X-axis: threshold value (0 to 1)
    - Y-axis: metric value (0 to 1)
    - Four lines: accuracy (blue), precision (green), recall (red), F1 (purple)
    - Vertical dashed line at threshold=0.5 (the default)
    - Optional: vertical line at the threshold that maximizes F1

    What this teaches
    -----------------
    The default threshold of 0.5 is arbitrary. This plot shows that:
    - Lowering the threshold: catches more positives (higher recall) but
      more false alarms (lower precision)
    - Raising the threshold: fewer false alarms (higher precision) but
      misses more positives (lower recall)
    - The right threshold depends on the cost of each error type.
    This is advanced logistic regression intuition that most tutorials skip.
    """
    self._check_fitted("plot_threshold_sensitivity")
    X, y = _validate_Xy(X, y)
    proba = self.predict_proba(X)

    thresholds = np.linspace(0.01, 0.99, 99)
    accuracies, precisions, recalls, f1s = [], [], [], []

    for t in thresholds:
        y_pred_t = (proba >= t).astype(int)
        TP = np.sum((y == 1) & (y_pred_t == 1))
        TN = np.sum((y == 0) & (y_pred_t == 0))
        FP = np.sum((y == 0) & (y_pred_t == 1))
        FN = np.sum((y == 1) & (y_pred_t == 0))
        n = len(y)

        acc = (TP + TN) / n
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    best_f1_idx = int(np.argmax(f1s))
    best_threshold = thresholds[best_f1_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, accuracies, color=FIT_LINE,     linewidth=2, label="Accuracy")
    ax.plot(thresholds, precisions, color=COLOR_RIDGE,  linewidth=2, label="Precision")
    ax.plot(thresholds, recalls,    color=RESIDUAL_POS, linewidth=2, label="Recall")
    ax.plot(thresholds, f1s,        color="#7047c4",    linewidth=2, label="F1 Score")

    ax.axvline(x=0.5, color="#aaaaaa", linewidth=1.2, linestyle="--",
               label="Default threshold (0.5)")
    ax.axvline(x=best_threshold, color="#7047c4", linewidth=1.2, linestyle=":",
               label=f"Best F1 threshold ({round(best_threshold, 2)})")

    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metric sensitivity to classification threshold")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if display:
        return self.show(fig=fig)
    return fig

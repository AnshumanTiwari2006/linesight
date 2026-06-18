import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy
from linesight.utils.colors import FIT_LINE
from linesight.utils.viz_context import print_viz_context


def plot_roc_curve(self, X, y, display: bool = True):
    """
    Plot the ROC (Receiver Operating Characteristic) curve.

    What is drawn
    -------------
    - X-axis: False Positive Rate (FPR) = FP / (FP + TN)
      = "of all actual negatives, what fraction did we incorrectly flag?"
    - Y-axis: True Positive Rate (TPR) = TP / (TP + FN)
      = "of all actual positives, what fraction did we correctly detect?"
    - The curve sweeps all possible thresholds from 0 to 1
    - Diagonal dashed line = random classifier (AUC = 0.5)
    - AUC (Area Under Curve) annotated on the plot

    What this teaches
    -----------------
    The ROC curve makes the FP/FN tradeoff visible: by choosing a lower
    threshold, you catch more positives (higher TPR) but also get more false
    alarms (higher FPR). The right threshold depends on your problem:
    - Medical diagnosis: prefer high TPR (catch every sick patient), accept FP
    - Spam filter: prefer low FPR (don't lose real emails), accept FN
    AUC = 1.0 is perfect. AUC = 0.5 is useless. AUC < 0.5 means the model
    is literally backwards.

    Algorithm
    ---------
    For 100 threshold values from 0 to 1:
        Apply threshold to predict_proba()
        Compute FPR and TPR at that threshold
    Plot FPR vs TPR.
    AUC = area under this curve, computed via trapezoidal rule.
    """
    self._check_fitted("plot_roc_curve")
    X, y = _validate_Xy(X, y)
    proba = self.predict_proba(X)

    thresholds = np.linspace(0, 1, 100)
    fprs = []
    tprs = []

    for t in thresholds:
        y_pred_t = (proba >= t).astype(int)
        TP = np.sum((y == 1) & (y_pred_t == 1))
        TN = np.sum((y == 0) & (y_pred_t == 0))
        FP = np.sum((y == 0) & (y_pred_t == 1))
        FN = np.sum((y == 1) & (y_pred_t == 0))

        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)

    # AUC via trapezoidal rule
    # Sort by FPR for correct integration
    sorted_pairs = sorted(zip(fprs, tprs))
    sorted_fprs = [p[0] for p in sorted_pairs]
    sorted_tprs = [p[1] for p in sorted_pairs]
    auc = float(np.trapezoid(sorted_tprs, sorted_fprs))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fprs, tprs, color=FIT_LINE, linewidth=2,
            label=f"ROC curve (AUC = {round(auc, 3)})")
    ax.plot([0, 1], [0, 1], color="#cccccc", linewidth=1.2,
            linestyle="--", label="Random classifier (AUC = 0.5)")

    # Mark the default 0.5 threshold point
    default_idx = np.argmin(np.abs(thresholds - 0.5))
    ax.scatter([fprs[default_idx]], [tprs[default_idx]],
               color="#e05252", s=80, zorder=5,
               label=f"Threshold=0.5 (FPR={round(fprs[default_idx],3)}, "
                     f"TPR={round(tprs[default_idx],3)})")

    ax.set_xlabel("False Positive Rate (FPR)\n'false alarm rate'")
    ax.set_ylabel("True Positive Rate (TPR)\n'sensitivity / recall'")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)

    print_viz_context(
        diagram="ROC Curve (Receiver Operating Characteristic)",
        solves="Evaluates the classifier's quality INDEPENDENT of any threshold choice.",
        theory=(
            "For every possible probability threshold t in [0,1], compute the TPR and FPR."
            " TPR (True Positive Rate / Recall) = fraction of actual positives correctly caught."
            " FPR (False Positive Rate) = fraction of actual negatives incorrectly labeled positive."
            " A perfect model: top-left corner (TPR=1, FPR=0). Random: the diagonal line."
            " AUC (Area Under Curve) summarizes performance in a single number."
        ),
        formula=(
            "TPR = TP / (TP + FN)   (= recall, sensitivity)\n"
            "FPR = FP / (FP + TN)   (= 1 - specificity)\n"
            "AUC = integral of TPR with respect to FPR  (trapezoidal rule)"
        ),
        constraints=(
            "- Requires predict_proba() to sweep thresholds (not just 0/1 predictions)\n"
            "- AUC is meaningful only for binary classification\n"
            "- AUC = 1.0 is perfect; AUC = 0.5 = random; AUC < 0.5 = reversed labels"
        ),
        reading=(
            "- Curve further top-left = better model overall.\n"
            "- Red dot = performance at the default 0.5 threshold.\n"
            "- Diagonal dashed line = a random coin-flip classifier.\n"
            "- Choose a point on the curve based on your FPR/TPR tolerance for your problem."
        ),
    )
    if display:
        return self.show(fig=fig)
    return fig

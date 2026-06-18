import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.validators import _validate_Xy


def plot_confusion_matrix(self, X, y, display: bool = True):
    """
    Plot a color-coded confusion matrix for classification results.

    Layout (2×2 grid):
    ---------------------------------
    | True Negative  | False Positive |
    | (TN)           | (FP)           |
    |----------------|----------------|
    | False Negative | True Positive  |
    | (FN)           | (TP)           |
    ---------------------------------

    Cell colors:
    - TN, TP (correct predictions): green shades
    - FP, FN (errors): red shades

    Each cell shows:
    - The count
    - What the error means in plain English:
        FP: "Predicted class 1, was actually class 0 (false alarm)"
        FN: "Predicted class 0, was actually class 1 (missed detection)"

    Also annotates: accuracy, precision, recall, F1 in a text box.

    What this teaches
    -----------------
    The confusion matrix is THE fundamental tool for understanding classification
    error. Students learn that accuracy alone is misleading (class imbalance),
    and that FP and FN have different real-world costs depending on the problem.
    """
    self._check_fitted("plot_confusion_matrix")
    X, y = _validate_Xy(X, y)
    y_pred = self.predict(X)

    # Compute matrix
    TP = int(np.sum((y == 1) & (y_pred == 1)))
    TN = int(np.sum((y == 0) & (y_pred == 0)))
    FP = int(np.sum((y == 0) & (y_pred == 1)))
    FN = int(np.sum((y == 1) & (y_pred == 0)))
    n = len(y)

    # Metrics
    accuracy  = (TP + TN) / n
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(7, 6))

    matrix = np.array([[TN, FP], [FN, TP]])
    labels = [["True Negative\n(TN)", "False Positive\n(FP)"],
              ["False Negative\n(FN)", "True Positive\n(TP)"]]
    descriptions = [
        ["Correct: predicted 0, was 0", "Error: predicted 1, was 0\n(false alarm)"],
        ["Error: predicted 0, was 1\n(missed)", "Correct: predicted 1, was 1"]
    ]
    cell_colors = [["#d4edda", "#f8d7da"], ["#f8d7da", "#d4edda"]]

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.axis('off')

    for i in range(2):
        for j in range(2):
            x_pos = j
            y_pos = 1 - i
            color = cell_colors[i][j]
            rect = plt.Rectangle([x_pos, y_pos], 1, 1,
                                  facecolor=color, edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_pos + 0.5, y_pos + 0.65, labels[i][j],
                    ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text(x_pos + 0.5, y_pos + 0.42, str(matrix[i][j]),
                    ha='center', va='center', fontsize=22, fontweight='bold')
            ax.text(x_pos + 0.5, y_pos + 0.15, descriptions[i][j],
                    ha='center', va='center', fontsize=7.5, color='#555555')

    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['Predicted: 0', 'Predicted: 1'], fontsize=11)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Actual: 1', 'Actual: 0'], fontsize=11)
    ax.tick_params(length=0)
    ax.xaxis.set_ticks_position('top')

    metrics_text = (f"Accuracy:  {round(accuracy, 4)}\n"
                    f"Precision: {round(precision, 4)}\n"
                    f"Recall:    {round(recall, 4)}\n"
                    f"F1 Score:  {round(f1, 4)}")
    ax.text(2.05, 1.0, metrics_text, ha='left', va='center',
            fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

    ax.set_title("Confusion Matrix", fontsize=13, pad=30)
    plt.tight_layout()

    if display:
        return self.show(fig=fig)
    return fig

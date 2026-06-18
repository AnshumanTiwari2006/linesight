import numpy as np
import matplotlib.pyplot as plt
from linesight.utils.viz_context import print_viz_context

def plot_residuals(self, X, y, display: bool = True):
    self._check_fitted("plot_residuals")
    # y must be 1D
    y = np.asarray(y).ravel()
    y_prob = self.predict_proba(X)
    residuals = y - y_prob
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_prob, residuals, color='#7047c4', edgecolors='white', linewidths=0.5, s=35, alpha=0.8)
    ax.axhline(0, color='#e05252', lw=1.5, ls='--')
    ax.set_title("plot_residuals() — Logistic Regression Residuals", fontsize=10, fontweight='bold')
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Residual (y - p)")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    print_viz_context(
        diagram="Logistic Regression Residuals Plot",
        purpose="Analyze the difference between actual binary outcomes and predicted probabilities.",
        theory="Residuals for logistic regression are defined as e = y - p, where p is the predicted probability. Plotting residuals against predicted probabilities helps assess whether there are systematic patterns in prediction errors.",
        formula="e_i = y_i - p_i",
        interpretation="Ideally, residuals should be randomly distributed. However, since y is binary, residuals will lie along two curves: one for y=1 (e = 1 - p) and one for y=0 (e = -p). Check for extreme outliers or gaps in probability range."
    )
    
    if display:
        return self.show(fig=fig)
    return fig

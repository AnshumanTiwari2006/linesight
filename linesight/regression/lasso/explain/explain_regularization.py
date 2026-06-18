import numpy as np


def explain_regularization(self) -> str:
    self._check_fitted("explain_regularization")
    alpha = self.alpha
    l1 = float(np.sum(np.abs(self.weights)))
    n_zero = int(np.sum(self.weights == 0))
    n_total = len(self.weights)

    lines = [
        "Lasso Regularization (L1)",
        "-" * 30,
        f"alpha = {alpha}",
        "",
        "Lasso adds alpha * sum(|theta|) to the MSE loss.",
        "Unlike Ridge (which uses theta^2), the absolute value creates SPARSITY:",
        "small coefficients get set to exactly zero, performing automatic feature selection.",
        "",
        f"Current sum of |feature coefficients|: {round(l1, 6)}",
        f"Features set to exactly zero: {n_zero} / {n_total}",
        "",
        "As alpha -> 0: Lasso approaches ordinary least squares.",
        "As alpha -> inf: All feature coefficients shrink to exactly 0.",
    ]
    output = "\n".join(lines)
    from linesight.utils.environment import _detect_environment
    if _detect_environment() == 'script':
        print(output)
    return output
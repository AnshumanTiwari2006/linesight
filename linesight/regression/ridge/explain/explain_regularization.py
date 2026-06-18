import numpy as np


def explain_regularization(self) -> str:
    """
    Print a plain-English explanation of what alpha (L2 regularization) is doing.
    Shows current alpha and the sum of squared coefficients.
    """
    self._check_fitted("explain_regularization")

    alpha = self.alpha
    l2_penalty = float(np.sum(self.weights ** 2))
    contribution = alpha * l2_penalty

    lines = [
        "Ridge Regularization (L2)",
        "-" * 30,
        f"alpha = {alpha}",
        f"",
        f"Ridge adds alpha * sum(theta^2) to the MSE loss.",
        f"This penalizes large coefficients and forces them to be small,",
        f"reducing overfitting by trading a little bias for lower variance.",
        f"",
        f"Current sum of squared feature coefficients: {round(l2_penalty, 6)}",
        f"Regularization contribution to loss: {round(contribution, 6)}",
        f"",
        f"Note: The intercept (theta[0]) is NEVER penalized.",
        f"Only feature weights (theta[1:]) are shrunk.",
        f"",
        f"As alpha -> 0: Ridge approaches ordinary least squares.",
        f"As alpha -> inf: All feature coefficients shrink toward 0.",
    ]

    output = "\n".join(lines)
    from linesight.utils.environment import _detect_environment
    if _detect_environment() == 'script':
        print(output)
    return output
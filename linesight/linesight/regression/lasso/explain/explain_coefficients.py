import numpy as np


def explain_coefficients(self) -> str:
    self._check_fitted("explain_coefficients")
    intercept = round(float(self.bias), 4)
    names = self.feature_names_in_

    lines = [
        f"Intercept: {intercept}",
        f"",
        f"Feature coefficients (L1-regularized, alpha={self.alpha}):",
        f"  Note: Coefficients at exactly 0 were eliminated by Lasso.",
        f"",
    ]
    for name, coef in zip(names, self.weights):
        coef_r = round(float(coef), 4)
        if coef_r == 0:
            lines.append(f"  {name}: 0 [eliminated]")
        else:
            direction = "increases" if coef_r > 0 else "decreases"
            lines.append(f"  {name}: {coef_r}")
            lines.append(f"    A 1-unit increase {direction} prediction by {abs(coef_r)}.")
        lines.append("")

    output = "\n".join(lines)
    from linesight.utils.environment import _detect_environment
    if _detect_environment() == 'script':
        print(output)
    return output
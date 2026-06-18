def explain_coefficients(self) -> str:
    self._check_fitted("explain_coefficients")
    intercept = round(float(self.bias), 4)
    names = self.feature_names_in_

    lines = [
        "Logistic Regression Coefficients (Log-Odds)",
        "-" * 43,
        f"Intercept: {intercept}",
        "  When all features are 0, the log-odds of class 1 is this intercept.",
        "",
    ]
    import numpy as np
    for name, coef in zip(names, self.weights):
        coef_r = round(float(coef), 4)
        odds_ratio = round(float(np.exp(coef)), 4)
        direction = "increases" if coef_r > 0 else "decreases"
        lines.append(f"Feature: {name} (coef: {coef_r})")
        lines.append(f"  A 1-unit increase in '{name}' {direction} the log-odds by {abs(coef_r)}.")
        lines.append(f"  This translates to multiplying the odds by {odds_ratio} (Odds Ratio).")
        lines.append("")

    output = "\n".join(lines)
    from linesight.utils.environment import _detect_environment
    if _detect_environment() == 'script':
        print(output)
    return output
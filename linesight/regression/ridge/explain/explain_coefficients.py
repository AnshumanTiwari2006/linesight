def explain_coefficients(self) -> str:
    """
    Print the Ridge coefficients with a note about L2 shrinkage.
    """
    self._check_fitted("explain_coefficients")

    intercept = round(float(self.bias), 4)
    names = self.feature_names_in_

    lines = [
        f"Intercept: {intercept}",
        f"  When all features are 0, the model predicts {intercept}.",
        f"",
        f"Feature coefficients (shrunk by L2 regularization, alpha={self.alpha}):",
        f"  Note: These are smaller than unregularized OLS coefficients would be.",
        f"",
    ]

    for name, coef in zip(names, self.weights):
        coef_r = round(float(coef), 4)
        direction = "increases" if coef_r > 0 else "decreases"
        lines.append(f"  {name}: {coef_r}")
        lines.append(
            f"    A 1-unit increase in '{name}' {direction} the prediction by {abs(coef_r)}."
        )
        lines.append("")

    output = "\n".join(lines)
    from linesight.utils.environment import _detect_environment
    if _detect_environment() == 'script':
        print(output)
    return output
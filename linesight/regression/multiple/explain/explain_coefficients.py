def explain_coefficients(self) -> str:
    """Print plain-English explanation of each learned coefficient."""
    self._check_fitted("explain_coefficients")

    intercept = round(float(self.bias), 4)
    coefs = self.weights
    names = self.feature_names_in_

    lines = [
        f"Intercept: {intercept}",
        f"  When all features are 0, the model predicts {intercept}.",
        "",
    ]

    for name, coef in zip(names, coefs):
        coef_r = round(float(coef), 4)
        direction = "increases" if coef_r > 0 else "decreases"
        lines.append(f"{name}: {coef_r}")
        lines.append(
            f"  Holding all other features constant, a 1-unit increase in '{name}' "
            f"{direction} the prediction by {abs(coef_r)}."
        )
        lines.append("")

    output = "\n".join(lines)
    from linesight.utils.environment import _detect_environment
    if _detect_environment() == 'script':
        print(output)
    return output
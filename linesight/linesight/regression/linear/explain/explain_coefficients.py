def explain_coefficients(self) -> str:
    """
    Print a plain-English explanation of each learned coefficient.

    Output example:
    ---------------
    Slope (coef_): 3.42
      For every 1-unit increase in the input,
      the predicted output increases by 3.42.

    Intercept: 1.20
      When the input is 0, the model predicts 1.20.

    Returns
    -------
    str — the full explanation (also printed to stdout)
    """
    self._check_fitted("explain_coefficients")

    m = round(float(self.m), 4)
    b = round(float(self.b), 4)

    direction = "increases" if m > 0 else "decreases"
    abs_m = abs(m)

    lines = [
        f"Slope (coef_): {m}",
        f"  For every 1-unit increase in the input,",
        f"  the predicted output {direction} by {abs_m}.",
        "",
        f"Intercept: {b}",
        f"  When the input is 0, the model predicts {b}.",
    ]

    output = "\n".join(lines)
    from linesight.utils.environment import _detect_environment
    if _detect_environment() == 'script':
        print(output)
    return output
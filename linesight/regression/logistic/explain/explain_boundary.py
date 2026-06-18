def explain_boundary(self) -> str:
    self._check_fitted("explain_boundary")
    intercept = round(float(self.bias), 4)
    coefs = [round(float(c), 4) for c in self.weights]
    names = self.feature_names_in_

    terms = " + ".join(f"{c}*{n}" for c, n in zip(coefs, names))
    boundary_eq = f"0 = {intercept} + {terms}"
    
    lines = [
        "Decision Boundary Explanation",
        "-" * 30,
        "The model predicts class 1 when the log-odds (z) > 0.",
        "The boundary between class 0 and class 1 is the line/plane where z = 0.",
        "",
        "Boundary Equation:",
        f"  {boundary_eq}",
    ]
    output = "\n".join(lines)
    from linesight.utils.environment import _detect_environment
    if _detect_environment() == 'script':
        print(output)
    return output
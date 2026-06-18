def explain_sigmoid(self) -> str:
    self._check_fitted("explain_sigmoid")
    lines = [
        "Sigmoid Function Explanation",
        "-" * 30,
        "The logistic regression model calculates a linear score (z = X * theta).",
        "It then passes z through the sigmoid function to get a probability:",
        "  p = 1 / (1 + exp(-z))",
        "",
        "Key properties:",
        "  - If z = 0, p = 0.5 (Decision boundary)",
        "  - If z > 0, p > 0.5 (Predicts class 1)",
        "  - If z < 0, p < 0.5 (Predicts class 0)",
        "  - As z approaches +inf, p approaches 1",
        "  - As z approaches -inf, p approaches 0",
    ]
    output = "\n".join(lines)
    from linesight.utils.environment import _detect_environment
    if _detect_environment() == 'script':
        print(output)
    return output
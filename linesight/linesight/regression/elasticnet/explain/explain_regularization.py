import numpy as np

def explain_regularization(self) -> str:
    """Explain ElasticNet L1/L2 balance."""
    self._check_fitted("explain_regularization")
    alpha = self.alpha
    l1_ratio = self.l1_ratio
    
    l1_penalty = float(np.sum(np.abs(self.weights)))
    l2_penalty = float(np.sum(self.weights ** 2))
    
    n_zero = int(np.sum(self.weights == 0))
    n_total = len(self.weights)
    
    lines = [
        "ElasticNet Regularization (L1 + L2)",
        "-" * 35,
        f"alpha = {alpha} (Overall regularization strength)",
        f"l1_ratio = {l1_ratio} (Balance between Lasso and Ridge)",
        "",
        "ElasticNet combines both L1 (Lasso) and L2 (Ridge) penalties:",
        "  - L1 creates sparsity (sets coefficients to exactly zero)",
        "  - L2 shrinks coefficients and handles correlated features better than L1 alone",
        "",
        f"L1 penalty sum(|theta|): {round(l1_penalty, 6)}",
        f"L2 penalty sum(theta^2): {round(l2_penalty, 6)}",
        f"Features set to exactly zero: {n_zero} / {n_total}",
    ]
    output = "\n".join(lines)
    from linesight.utils.environment import _detect_environment
    if _detect_environment() == 'script':
        print(output)
    return output
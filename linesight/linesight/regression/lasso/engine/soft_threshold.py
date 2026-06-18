def _soft_threshold(rho: float, alpha: float) -> float:
    """
    The Lasso coordinate descent update operator. The mathematical heart of Lasso.

    Shrinks rho toward zero by alpha. If |rho| <= alpha, returns exactly 0.
    This is why Lasso creates SPARSITY — small coefficients get zeroed exactly,
    unlike Ridge which only makes them small.

    Derivation: Lasso adds |weight_j| to the loss. The subgradient at weight_j=0
    is [-alpha, alpha]. The coordinate descent update minimizing this has the
    closed form below.

    Parameters
    ----------
    rho : float — the unpenalized optimal value for this coordinate
    alpha : float — regularization strength

    Returns
    -------
    rho - alpha   if rho > alpha  (positive, but reduced)
    rho + alpha   if rho < -alpha (negative, but reduced)
    0             if |rho| <= alpha (zeroed out — creates sparsity)
    """
    if rho > alpha:
        return rho - alpha
    elif rho < -alpha:
        return rho + alpha
    else:
        return 0.0

def _soft_threshold(rho: float, alpha: float) -> float:
    """Soft-threshold operator for coordinate descent (same as Lasso)."""
    if rho > alpha:
        return rho - alpha
    elif rho < -alpha:
        return rho + alpha
    else:
        return 0.0

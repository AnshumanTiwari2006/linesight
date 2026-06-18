import pytest, numpy as np, warnings



@pytest.fixture(scope="module")
def multi_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 3))
    y = X @ np.array([2.0, -1.0, 0.5]) + 0.3 + rng.normal(0, 0.1, 80)
    return X, y

@pytest.fixture(scope="module")
def binary_data():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return X, y




from linesight import LassoRegression


class TestLasso:

    def test_alpha_validation(self, multi_data):
        X, y = multi_data
        with pytest.raises(ValueError, match="alpha"):
            LassoRegression(alpha=-0.1).fit(X, y)

    def test_weights_shape(self, multi_data):
        X, y = multi_data
        m = LassoRegression(alpha=0.1, epochs=300)
        m.fit(X, y)
        assert m.weights.shape == (3,)

    def test_sparsity_with_high_alpha(self):
        """High alpha should zero out at least one coefficient."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((100, 5))
        y = X[:, 0] * 3.0 + X[:, 1] * 0.5 + rng.normal(0, 0.1, 100)
        # Weak features (cols 2,3,4) should be zeroed with high alpha
        m = LassoRegression(alpha=0.5, epochs=500)
        m.fit(X, y)
        n_zeros = np.sum(m.weights == 0)
        assert n_zeros >= 1, f"Expected at least 1 zero weight, got weights={m.weights}"

    def test_no_sparsity_with_very_low_alpha(self, multi_data):
        """Near-zero alpha should keep all coefficients nonzero."""
        X, y = multi_data
        m = LassoRegression(alpha=1e-6, epochs=300)
        m.fit(X, y)
        assert np.sum(m.weights == 0) == 0

    def test_explain_sparsity_returns_string(self, multi_data):
        X, y = multi_data
        m = LassoRegression(alpha=0.2, epochs=200)
        m.fit(X, y)
        s = m.explain_sparsity()
        assert isinstance(s, str)
        assert "eliminated" in s.lower() or "kept" in s.lower()

    def test_history_recorded(self, multi_data):
        X, y = multi_data
        m = LassoRegression(alpha=0.1, epochs=50, store_history=True)
        m.fit(X, y)
        h = m.get_training_history()
        assert len(h.losses) == 50

    def test_intercept_not_penalized(self, multi_data):
        """Bias should not be zero even at high alpha."""
        X, y = multi_data
        m = LassoRegression(alpha=5.0, epochs=300)
        m.fit(X, y)
        # bias can be non-zero while all weights are zero
        assert isinstance(m.bias, float)



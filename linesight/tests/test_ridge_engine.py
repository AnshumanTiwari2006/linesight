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




from linesight import RidgeRegression


class TestRidge:

    def test_alpha_validation(self, multi_data):
        X, y = multi_data
        with pytest.raises(ValueError, match="alpha"):
            RidgeRegression(alpha=-0.1).fit(X, y)

    def test_weights_shape(self, multi_data):
        X, y = multi_data
        m = RidgeRegression(alpha=0.1, epochs=500, normalize=True)
        m.fit(X, y)
        assert m.weights.shape == (3,)

    def test_r2_reasonable(self, multi_data):
        X, y = multi_data
        m = RidgeRegression(alpha=0.01, epochs=1000, normalize=True)
        m.fit(X, y)
        assert m.score(X, y)["r2"] > 0.85

    def test_higher_alpha_smaller_weights(self, multi_data):
        """Stronger regularization should produce smaller weight magnitudes."""
        X, y = multi_data
        m_low  = RidgeRegression(alpha=0.001, epochs=800, normalize=True)
        m_high = RidgeRegression(alpha=10.0,  epochs=800, normalize=True)
        m_low.fit(X, y)
        m_high.fit(X, y)
        assert np.sum(np.abs(m_high.weights)) <= np.sum(np.abs(m_low.weights))

    def test_history_recorded(self, multi_data):
        X, y = multi_data
        m = RidgeRegression(alpha=0.1, epochs=100, store_history=True)
        m.fit(X, y)
        h = m.get_training_history()
        assert not h.is_empty()
        assert len(h.losses) == 100

    def test_explain_regularization_returns_string(self, multi_data):
        X, y = multi_data
        m = RidgeRegression(alpha=1.0, epochs=100)
        m.fit(X, y)
        s = m.explain_regularization()
        assert isinstance(s, str)
        assert "alpha" in s.lower() or "ridge" in s.lower()

    def test_coef_and_intercept_aliases(self, multi_data):
        X, y = multi_data
        m = RidgeRegression(alpha=0.1, epochs=200)
        m.fit(X, y)
        np.testing.assert_array_equal(m.coef_, m.weights)
        assert m.intercept_ == m.bias



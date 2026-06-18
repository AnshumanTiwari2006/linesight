import pytest, numpy as np, warnings

@pytest.fixture(scope="module")
def multi_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 3))
    y = X @ np.array([2.0, -1.0, 0.5]) + 0.3 + rng.normal(0, 0.1, 80)
    return X, y

from linesight import ElasticNetRegression

class TestElasticNet:

    def test_alpha_validation(self, multi_data):
        X, y = multi_data
        with pytest.raises(ValueError, match="alpha"):
            ElasticNetRegression(alpha=-0.1).fit(X, y)

    def test_l1_ratio_validation(self, multi_data):
        X, y = multi_data
        with pytest.raises(ValueError, match="l1_ratio"):
            ElasticNetRegression(l1_ratio=1.5).fit(X, y)

    def test_fit_and_weights_shape(self, multi_data):
        X, y = multi_data
        m = ElasticNetRegression(alpha=0.1, l1_ratio=0.5, epochs=300)
        m.fit(X, y)
        assert m.weights.shape == (3,)

    def test_r2_reasonable(self, multi_data):
        X, y = multi_data
        m = ElasticNetRegression(alpha=0.01, l1_ratio=0.5, epochs=500)
        m.fit(X, y)
        assert m.score(X, y)["r2"] > 0.85

    def test_sparsity_with_high_alpha_and_l1_ratio(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((100, 5))
        y = X[:, 0] * 3.0 + X[:, 1] * 0.5 + rng.normal(0, 0.1, 100)
        
        m = ElasticNetRegression(alpha=1.0, l1_ratio=0.9, epochs=300)
        m.fit(X, y)
        assert np.any(np.abs(m.weights) < 1e-6)

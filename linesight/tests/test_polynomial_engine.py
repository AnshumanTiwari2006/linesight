import pytest, numpy as np, warnings

@pytest.fixture(scope="module")
def poly_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 1))
    # y = 2x^2 - x + 1
    y = 2 * (X[:, 0]**2) - X[:, 0] + 1 + rng.normal(0, 0.1, 100)
    return X, y

from linesight import PolynomialRegression
from linesight.exceptions import LineSightShapeError

class TestPolynomial:

    def test_degree_validation(self, poly_data):
        X, y = poly_data
        with pytest.raises(ValueError, match="degree"):
            PolynomialRegression(degree=0).fit(X, y)

    def test_fit_and_weights_shape(self, poly_data):
        X, y = poly_data
        m = PolynomialRegression(degree=2, epochs=500, normalize=True)
        m.fit(X, y)
        assert m.theta_.shape == (3,)

    def test_predict_and_score(self, poly_data):
        X, y = poly_data
        m = PolynomialRegression(degree=2, epochs=500, normalize=True)
        m.fit(X, y)
        score = m.score(X, y)
        assert score["r2"] > 0.6

    def test_multi_feature_raises_error(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 2))
        y = X[:, 0] + X[:, 1]
        m = PolynomialRegression(degree=2)
        with pytest.raises(LineSightShapeError):
            m.fit(X, y)

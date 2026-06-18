from linesight.regression.linear.core import LinearRegression
from linesight.regression.polynomial.core import PolynomialRegression
from linesight.regression.multiple.core import MultipleLinearRegression
from linesight.regression.ridge.core import RidgeRegression
from linesight.regression.lasso.core import LassoRegression
from linesight.regression.elasticnet.core import ElasticNetRegression
from linesight.regression.logistic.core import LogisticRegression

__version__ = "0.1.0"

__all__ = [
    "LinearRegression",
    "PolynomialRegression",
    "MultipleLinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "ElasticNetRegression",
    "LogisticRegression",
]


import numpy as np
import os

from kaolin import Surrogate
from sklearn.utils.testing import assert_almost_equal


def f(x):
    return x * np.sin(x)


X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
X2 = np.atleast_2d([2., 4., 5.5, 6.5, 7.5]).T
y = f(X).ravel()


def test_interpolation():
    surrogate = Surrogate().fit(X, y)
    y_pred, y_cov = surrogate.predict(X, return_cov=True)

    assert_almost_equal(y_pred, y)
    assert_almost_equal(np.diag(y_cov), 0.)


def test_multioutput():
    y_2d = np.vstack((y, y * 2)).T

    surrogate = Surrogate()
    surrogate.fit(X, y)

    surrogate_2d = Surrogate()
    surrogate_2d.fit(X, y_2d)

    y_pred_1d, y_std_1d = surrogate.predict(X2, return_std=True)
    y_pred_2d, y_std_2d = surrogate_2d.predict(X2, return_std=True)
    _, y_cov_1d = surrogate.predict(X2, return_cov=True)
    _, y_cov_2d = surrogate_2d.predict(X2, return_cov=True)

    assert_almost_equal(y_pred_1d, y_pred_2d[:, 0])
    assert_almost_equal(y_pred_1d, y_pred_2d[:, 1] / 2)

    # Standard deviation and covariance do not depend on output
    assert_almost_equal(y_std_1d, y_std_2d)
    assert_almost_equal(y_cov_1d, y_cov_2d)


def test_save_load():
    filename = 'tmp.kaolin'
    surrogate = Surrogate().fit(X, y)
    surrogate.save(filename)
    surrogate = None

    surrogate = Surrogate().load(filename)
    y_pred, y_cov = surrogate.predict(X, return_cov=True)

    assert_almost_equal(y_pred, y)
    assert_almost_equal(np.diag(y_cov), 0.)

    try:
        os.remove(filename)
    except OSError:
        pass


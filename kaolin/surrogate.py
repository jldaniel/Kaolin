

# Can generate a lot of UserWarnings, suppress these
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.model_selection import GridSearchCV
from sklearn.utils import check_X_y


# TODO Also explore kernel combinations using Sum and Product
# Kernels
ker_rbf = ConstantKernel(1.0, constant_value_bounds="fixed") * \
          RBF(1.0, length_scale_bounds="fixed")

ker_rq = ConstantKernel(1.0, constant_value_bounds="fixed") * \
         RationalQuadratic(alpha=0.1, length_scale=1)

ker_expsine = ConstantKernel(1.0, constant_value_bounds="fixed") * \
              ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1))

ker_mat = ConstantKernel(1.0, constant_value_bounds="fixed") * \
          Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)

kernels = [ker_rbf, ker_rq, ker_expsine, ker_mat]


class Surrogate(BaseEstimator, RegressorMixin):
    # TODO: Experiment with using separate regressors for each output
    def __init__(self, random_state=None):
        self.random_state = random_state

        self.X_train_ = None
        self.y_train_ = None
        self.kernel_ = None
        self.best_estimator_ = None
        self.cv_results_ = None

    def fit(self, X, y=None):
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        self.X_train_ = np.copy(X)
        self.y_train_ = np.copy(y)

        gpr = GaussianProcessRegressor(
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=10,
            random_state=self.random_state
        )

        param_grid = {"kernel": kernels, "alpha": [1e-10]}

        grid_search = GridSearchCV(
            gpr,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            return_train_score=True
        )

        grid_search.fit(X, y)

        self.best_estimator_ = grid_search.best_estimator_
        self.cv_results_ = grid_search.cv_results_

        return self

    def predict(self, X, return_std=False):
        return self.best_estimator_.predict(X, return_std=return_std)

    def save(self, location):
        pass
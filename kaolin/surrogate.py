

# Can generate a lot of UserWarnings, suppress these
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import copy
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.model_selection import GridSearchCV
from sklearn.utils import check_X_y
from sklearn.model_selection import cross_validate
import time

from kaolin.utils import mixed_doe

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
    def __init__(self, random_state=None, n_proc=1):
        self.random_state = random_state

        self.X_train_ = None
        self.y_train_ = None
        self.kernel_ = None
        self.best_estimator_ = None
        self.cv_results_ = None
        self.n_proc = n_proc
        self.cv_metrics = {"explained_variance": None,
                           "mean_absolute_error": None,
                           "mean_squared_error": None,
                           "r2_score": None,
                           "training_time": None}

    def fit(self, X, y=None):
        start_train_time = time.time()
        # TODO Check for repeated values and remove them, raise a Warning
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        self.X_train_ = np.copy(X)
        self.y_train_ = np.copy(y)

        gpr = GaussianProcessRegressor(
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=10,
            random_state=self.random_state
        )

        param_grid = {"kernel": kernels, "alpha": [1e-10, 1e-5]}

        grid_search = GridSearchCV(
            gpr,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            return_train_score=True,
            n_jobs=self.n_proc
        )

        grid_search.fit(X, y)

        self.best_estimator_ = grid_search.best_estimator_
        self.cv_results_ = grid_search.cv_results_
        stop_train_time = time.time()

        # Calculate the cross validation metrics
        scoring = ['explained_variance',
                   'neg_mean_absolute_error',
                   'neg_mean_squared_error',
                   'r2']

        scores = cross_validate(self.best_estimator_,
                                self.X_train_,
                                self.y_train_,
                                scoring=scoring, cv=5)

        self.cv_metrics['explained_variance'] = np.mean(scores['test_explained_variance'])
        self.cv_metrics['mean_absolute_error'] = -np.mean(scores['test_neg_mean_absolute_error'])
        self.cv_metrics['mean_squared_error'] = -np.mean(scores['test_neg_mean_squared_error'])
        self.cv_metrics['r2_score'] = np.mean(scores['test_r2'])
        self.cv_metrics['training_time'] = 1000*(stop_train_time - start_train_time)

        return self

    def predict(self, X, return_std=False, return_cov=False):
        """
        Predict the response at the given sites X
        :param X: array like sites to predict the response at
        :param return_std: Set to true to return the standard deviation at the prediction sites
        :param return_cov: Set to true to return the covariance at the prediction sites
        :return: array like set of predictions
        """
        return self.best_estimator_.predict(X, return_std=return_std, return_cov=return_cov)

    def adapt(self, bounds, n_points=1):
        """
        Find n_points designs that will best improve the surrogate model for the given bounds. If the surrogate
        has not been trained, will produce a suitable design of experiments to use for the initial training set.
        :param n_points:
        :return:
        """

        if n_points < 0:
            raise ValueError('The given number of adaption points ' + repr(n_points) + ' must be greater than 0')

        designs = []
        if self.best_estimator_ is None:
            # No estimator has been trained yet, so generate an initial DOE
            designs = mixed_doe(n_points, bounds, seed=self.random_state)
            return designs

        model = copy.deepcopy(self.best_estimator_)
        n_dim = bounds.shape[0]

        design = differential_evolution(func=Surrogate._model_error,
                                        bounds=bounds,
                                        popsize=10*n_dim,
                                        init='random',
                                        maxiter=50,
                                        tol=1e-5,
                                        args=model)

        designs.append(design.x)

        if n_points == 1:
            return designs

        # TODO Parallelize this process across n_proc processes
        for i in range(1, n_points):
            # Update the model assuming that the predicted point is true
            X_train = np.concatenate((self.X_train_, designs[i-1]), axis=0)
            y_train = np.concatenate((self.y_train_, model.predict(designs[i-1])))

            model.fit(X_train, y_train)

            # Find the next design
            design = differential_evolution(func=Surrogate._model_error,
                                            bounds=bounds,
                                            popsize=10 * n_dim,
                                            init='random',
                                            maxiter=50,
                                            tol=1e-5,
                                            args=model)

            designs.append(design.x)

        return designs

    def improve(self, X, y):
        """
        Improve the surrogate model with additional data
        :param X: Input training data
        :param y: Output training data
        :return: self
        """
        if self.best_estimator_ is None:
            self.fit(X, y)
        else:
            X_train = np.concatenate((self.X_train_, X), axis=0)
            y_train = np.concatenate((self.y_train_, y))

            self.fit(X_train, y_train)

        return self

    def info(self):
        """
        Display info about the surrogate model
        :return: None
        """
        pass

    def save(self, location):
        pass

    @staticmethod
    def _model_error(self, X, model):
        n_dim = model.X_train_.shape[1]
        _, std_dev = model.predict(X.reshape(-1, n_dim), return_std=True)
        return std_dev
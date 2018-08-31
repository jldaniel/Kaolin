
from kaolin import Surrogate
from kaolin.utils import latin_hypercube
from touchstone.models import Ackley, Rosenbrock
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score

import time

# Goal: Press run to generate a set of benchmark data and a report that can be used to compare
#       to a previous benchmark run to easily evaluate tweaks to the algorithm.

# TODO Move metric calculation to the Surrogate
# TODO Create a benchmark class, Benchmark(model)
# TODO Move benchmark results to a DataFrame for easy comparison and analysis
# TODO Generate benchmark results report
# TODO Generate plots for visual comparison

n_train = 20
n_test = 100

# Benchmark the Ackley model
model = Ackley()
surrogate = Surrogate(random_state=42, n_proc=4)
x_train = surrogate.adapt(model.bounds, n_train)
y_train = model(x_train)
tic = time.time()
surrogate.fit(x_train, y_train)
toc = time.time()

x_test = latin_hypercube(model.bounds, n_test, seed=42)
y_true = model(x_test)
y_pred = surrogate.predict(x_test)

# Regression Metrics
print('=====  RUN 1 ======')
print('n_train: ' + repr(n_train))
print('n_test: ' + repr(n_test))
print('Regression Metrics')
print('Explained Variance: ' + repr(explained_variance_score(y_true, y_pred)))
print('Mean Absolute Error: ' + repr(mean_absolute_error(y_true, y_pred)))
print('Mean Squared Error: ' + repr(mean_squared_error(y_true, y_pred)))
print('Mean Squared Log Error: ' + repr(mean_squared_log_error(y_true, y_pred)))
print('Medain Absolute Error: ' + repr(median_absolute_error(y_true, y_pred)))
print('R^2 Score: ' + repr(r2_score(y_true, y_pred)))
print('Training Time: ' + repr(1000*(toc - tic)) + "ms")


n_train = 200
n_test = 100

# Benchmark the Ackley model
model = Ackley()
surrogate = Surrogate(random_state=42, n_proc=4)
x_train = surrogate.adapt(model.bounds, n_train)
y_train = model(x_train)
tic = time.time()
surrogate.fit(x_train, y_train)
toc = time.time()

x_test = latin_hypercube(model.bounds, n_test, seed=42)
y_true = model(x_test)
y_pred = surrogate.predict(x_test)

# Regression Metrics
print("===== RUN 2 ======")
print('n_train: ' + repr(n_train))
print('n_test: ' + repr(n_test))
print('Regression Metrics')
print('Explained Variance: ' + repr(explained_variance_score(y_true, y_pred)))
print('Mean Absolute Error: ' + repr(mean_absolute_error(y_true, y_pred)))
print('Mean Squared Error: ' + repr(mean_squared_error(y_true, y_pred)))
print('Mean Squared Log Error: ' + repr(mean_squared_log_error(y_true, y_pred)))
print('Medain Absolute Error: ' + repr(median_absolute_error(y_true, y_pred)))
print('R^2 Score: ' + repr(r2_score(y_true, y_pred)))
print('Training Time: ' + repr(1000*(toc - tic)) + "ms")
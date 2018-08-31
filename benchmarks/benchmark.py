
from kaolin import Surrogate
from kaolin.utils import latin_hypercube
from touchstone.models import Ackley, Booth, Bukin, CrossInTray, DropWave, Eggholder, GramacyLee, HolderTable, Langermann, Rosenbrock
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

import time

# Goal: Press run to generate a set of benchmark data and a report that can be used to compare
#       to a previous benchmark run to easily evaluate tweaks to the algorithm.

# TODO Move benchmark results to a DataFrame for easy comparison and analysis
# TODO Generate benchmark results report
# TODO Generate plots for visual comparison


class Benchmark(object):
    def __init__(self, models, n_train, n_test, random_state=None, n_proc=1):
        self.models = models
        self.n_train = n_train
        self.n_test = n_test
        self.random_state = random_state
        self.n_proc = n_proc

    def run(self):
        print('Starting Benchmark...')
        print('\tn_train: ' + repr(self.n_train))
        print('\tn_test: ' + repr(self.n_test))

        for model in self.models:
            print('\n=== ' + model.name + '===')
            surrogate = Surrogate(random_state=self.random_state, n_proc=self.n_proc)
            x_train = surrogate.adapt(model.bounds, self.n_train)
            y_train = model(x_train)
            tic = time.time()
            surrogate.fit(x_train, y_train)
            toc = time.time()

            x_test = latin_hypercube(model.bounds, self.n_test, seed=42)
            y_true = model(x_test)
            y_pred = surrogate.predict(x_test)

            print('Results: ')
            print('Training Time: ' + repr(1000*(toc - tic)) + "ms")

            print('\nCross Validation Metrics: ')
            print('\tExplained Variance: ' + repr(surrogate.cv_metrics['explained_variance']))
            print('\tMean Absolute Error: ' + repr(surrogate.cv_metrics['mean_absolute_error']))
            print('\tMean Squared Error: ' + repr(surrogate.cv_metrics['mean_squared_error']))
            print('\tR^2 Score: ' + repr(surrogate.cv_metrics['r2_score']))

            print('\nTest Metrics')
            print('\tExplained Variance: ' + repr(explained_variance_score(y_true, y_pred)))
            print('\tMean Absolute Error: ' + repr(mean_absolute_error(y_true, y_pred)))
            print('\tMean Squared Error: ' + repr(mean_squared_error(y_true, y_pred)))
            print('\tR^2 Score: ' + repr(r2_score(y_true, y_pred)))


if __name__ == '__main__':
    n_train = 20
    n_test = 100
    models = [Ackley(),
              Booth(),
              Bukin(),
              CrossInTray(),
              DropWave(),
              Eggholder(),
              GramacyLee(),
              HolderTable(),
              Langermann(),
              Rosenbrock()]
    benchmark = Benchmark(models, n_train, n_test, random_state=42, n_proc=4)
    benchmark.run()

[![Build Status](https://travis-ci.org/jldaniel/Kaolin.svg?branch=master)](https://travis-ci.org/jldaniel/Kaolin)

# Kaolin

ka·o·lin
/ˈkāələn/

_noun_

a fine, soft white clay, resulting from the natural decomposition of other clays or feldspar. It is used for making porcelain and china, as a filler in paper and textiles, and in medicinal absorbents.


#### About

Kaolin is a package for generating gaussian process surrogate models particularly when the surrogate is going to be used in an automated process or when one does not want to spend time hand tuning the model. Kaolin achieves this by automatically performing a search for a suitable kernel during the fit process and contains some helpful utilities for generating training sets and adapting the model.

#### Usage

Example of using Kaolin

__Basic Usage__

```python

from touchstone.models import Rosenbrock
from kaolin import Surrogate

# Create the true model to use to generate training data
model = Rosenbrock()

# Create the surrogate
surrogate = Surrogate()

# Create a training set design of experiments by calling the adapt
# method which will generate an initial DOE if the model has not been
# trained yet
n_train = 20
x_train = surrogate.adapt(model.bounds, n_train)
y_train = model(x_train)

# Train the surrogate
surrogate.fit(x_train, y_train)

# Make a prediction
x_pred = [[1.23, 3.42], [0.15, 1.2]]
y_pred, std_dev = surrogate.predict(x_pred, return_std=True)


# Improve the surrogate
n_improve = 10
x_improve = surrogate.adapt(model.bounds, n_improve)
y_improve = model(x_improve)
surrogate.improve(x_improve, y_improve)

# Save the model
surrogate.save('rosenbrock.kaolin')

```
# Roulette - more than a metric.

__Roulette__ is a unified way to evaluate Machine Learning models. At its core, Roulette is using a [Monte Carlo Simulation](https://en.wikipedia.org/wiki/Monte_Carlo_method) (=MCS) to estimate the risks in deploying a ML model to real world. The results of the MCS are aggregated using [Wasserstein Distance](https://en.wikipedia.org/wiki/Wasserstein_metric) (=WD) and result with two metrics:

1. Distinguishability: a measure of accuracy = by how much the model is better than the data-mid-point: mean / most common value. value is in the range [0,1]

2. Certainty: a measure that of consistency = by how much the model prediction are consistent over different samples of the data. value is >1, higher is better.


## Installing

Roulette is hosted on PyPi, install using pip

```pip install Roulette```

## Usage 

We demonstrate the use of regression builder, binary calssification is reletavely similar.

### Loading data

Roulette works with a single dataframe, with all the features and the target.

```python
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
import seaborn as sns
import pandas as pd
import numpy as np

boston = load_boston()
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data['PRICE'] = boston.target
```

### Loading __Roulette__

```python
builder = RegressionBuilder(
    "path_to_model_file",
    data,
    "PRICE",
    "mse",
    "min_max"
)
```

### Building model

```python
builder.build(n_experiments=1000)
builder.result # will return a dictionary {'discriminability': 0.8840, 'certainty': 8.245}
builder.plot()
builder.finalize_model() # runs a model build on the entire dataset
builder.save() # will create a local artifact on 'path_to_model_file/builder'
```
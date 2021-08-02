# Causal Influence Measures Estimation
Causal tools to analyze real datasets 

The `causalestimation` module contains a tool for estimating a causal distribution from an observational data set under presence of confounders using kernelization procedure. You can supply an set of "confounders" variables for controlling, and measure either the expected value of the effect given the cause or the informational version called local information flow (see the file for the definition) of that.

```python
import numpy as np
import pandas as pd

#Generating some toy causal model with confounders 

reps = 2000
Z = np.random.normal(size=reps)
X1 = Z + np.random.normal(size=reps)
X2 = Z + np.random.normal(size=reps)
Y = X1 + X2 + np.random.normal(size=reps)

# load the data into a dataframe:
data = pd.DataFrame({'Z' : Z, 'X1' : X1, 'X2' : X2, 'Y' : Y})

# define the variable types: 'c' is 'continuous'.  The variables defined here
# are the ones the search is performed over  -- NOT all the variables defined
# in the data frame.

types = {'Z' : 'c', 'X1' : 'c', 'X2' : 'c', 'Y' : 'c'}

# Estimating effect

from causalestimation import CausalEffect

CE = CausalEffect(data, ['X1'], ['Y'], confounders=['Z'], variable_types=types)
```

You can see the averaged treatment/causal effect (ATE) of intervention, `E_{x1}[P(Y|do(x1))]` using the measured causal effect with `confounders=['Z']`,
```python
>>> x1 = np.mean(CE.support['X1'])
>>> x = pd.DataFrame({'X1' : [x1]})
>>> print(CE.ATE(x))
```

For an informational perspective you can type, `info=True` to get the local flow [2] from `x1` to the effect `['Y']` in the presence of `['Z']`,
```python

CE = CausalEffect(data, ['X1'], ['Y'], confounders=['Z'], variable_types=types, info=True)
CE.local_flow(x)
```
And, the averaged local flow (or information flow [1]) from `[X1]` to `['Y']` is given by:
```python

args = CE.causes
CE.local_flow(args)
```

This repository is in its first steps waiting for publishing the whole analysis with data.\
[1] https://doi.org/10.1142/S0219525908001465 \
[2] https://doi.org/10.3390/e22080854

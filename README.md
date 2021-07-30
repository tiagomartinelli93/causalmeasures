# Causal Influence Measures Estimation
Causal tools to analyze real datasets 

The `causalestimation` module contains a tool for estimating a causal distribution from an observational data set under presence of confounders using kernelization procedure. You can supply an set of "confounders" variables for controlling, and measure either the expected value of the effect given the cause or the informational version called local information flow (see the file for the definition) of that.

```python
import numpy as np
import pandas as pd

#Generating some toy causal model with confounders 

reps = 2000
x1 = np.random.normal(size=reps)
x2 = x1 + np.random.normal(size=reps)
x3 = x1 + np.random.normal(size=reps)
x4 = x2 + x3 + np.random.normal(size=reps)
x5 = x4 + np.random.normal(size=reps)

# load the data into a dataframe:
data = pd.DataFrame({'x1' : x1, 'x2' : x2, 'x3' : x3, 'x4' : x4, 'x5' : x5})

# define the variable types: 'c' is 'continuous'.  The variables defined here
# are the ones the search is performed over  -- NOT all the variables defined
# in the data frame.

types = {'x1' : 'c', 'x2' : 'c', 'x3' : 'c', 'x4' : 'c', 'x5' : 'c'}

# Estimating effect

from causalestimation import CausalEffect

x = pd.DataFrame({'x2' : [0.], 'x3' : [0.]})
CE = CausalEffect(data, ['x2'], ['x3'], confounders=['x1'], variable_types=types)
CE
```

You can see the causal effect of intervention, `P(x3|do(x2))` using the measured causal effect in `confounders`,
```python
>>> x = pd.DataFrame({'x2' : [0.], 'x3' : [0.]})
>>> print(CE.expected_value(x))
0.268915603296
```

This repository is in its first steps waiting for publishing the whole analysis with data. We adapted some function nonparametric estimation from https://github.com/akelleh/causality

# Causal Influence Measures Estimation
Causal tools to analyze real datasets 

The `nonparametric` module contains a tool for non-parametrically estimating a causal distribution from an observational data set under presence of confounders. You can supply an set of "confounders" variables for controlling, and the measure either the expected value of the effect given the cause or the informationa version called local information flow (see the file for the definition).

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

from nonparametric import CausalEffect

x = pd.DataFrame({'x2' : [0.], 'x3' : [0.]})
CE = CausalEffect(data, ['x2'], ['x3'], admissable_set=['x1'], variable_types=types)
CE
```

You can see the causal effect of intervention, `P(x3|do(x2))` using the measured causal effect in `adjustment`,
```python
>>> x = pd.DataFrame({'x2' : [0.], 'x3' : [0.]})
>>> print(CE.expected_value(x))
0.268915603296
```

Which is close to the correct value of `0.282` for a gaussian with mean 0. and variance 2.  If you adjust the value of `'x2'`, you'll find that the probability of `'x3'` doesn't change.  This is untrue with just the conditional distribution, `P(x3|x2)`, since in this case, observation and intervention are not equivalent.

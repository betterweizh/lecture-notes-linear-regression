# Quantile Regression in Python

## Overview

This chapter fits the linear quantile regression model of
{doc}`22-quantile-regression` with `scikit-learn` and `statsmodels`, compares
their estimates, and fits several quantiles at once to describe how the
whole conditional distribution depends on a predictor. It applies the methods of {doc}`22-quantile-regression`. The code was tested with `statsmodels` 0.14 and `scikit-learn` 1.9.

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- fit linear quantile regressions with `QuantileRegressor` and `QuantReg`;
- switch off the default penalty of `QuantileRegressor`; and
- fit and compare several quantiles.
:::

## Fitting one quantile

Two libraries provide linear quantile regression:

- `scikit-learn`:
  [`sklearn.linear_model.QuantileRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html);
- `statsmodels`:
  [`statsmodels.regression.quantile_regression.QuantReg`](https://www.statsmodels.org/stable/generated/statsmodels.regression.quantile_regression.QuantReg.html).

The following example fits the $0.1$-quantile with both libraries on the
same simulated data, so that the estimates can be compared.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import QuantileRegressor

# Simulated data
rng = np.random.default_rng(0)
x = np.linspace(0, 10, 100) + (1 - rng.random(100)) * 5
X = x.reshape(-1, 1)                      # predictors as a 2-D array
y = 2 * x + (1 - rng.random(100)) * 5     # response as a 1-D array

# scikit-learn: alpha=0 switches off the default L1 penalty
model = QuantileRegressor(quantile=0.1, alpha=0.0, fit_intercept=True)
model.fit(X, y)
print(model.intercept_, model.coef_)

# statsmodels: add the intercept column explicitly
res = sm.QuantReg(endog=y, exog=sm.add_constant(X)).fit(q=0.1)
print(res.params)
```

`QuantileRegressor` adds an $L_1$ penalty with weight `alpha` (default
$1.0$); setting `alpha=0.0` gives the unpenalized estimator
{eq}`eq-quantile-regression`. `QuantReg` reports standard errors and
confidence intervals in `res.summary()`.

Both libraries give the same estimates up to the solver tolerance
(intercept about $0.3205$ and slope about $2.0118$ for this simulated
sample). By default `QuantileRegressor` uses the HiGHS linear programming
solver (`solver="highs"`).

## Fitting several quantiles

```python
taus = [0.1, 0.25, 0.5, 0.75, 0.9]
fits = {tau: sm.QuantReg(y, sm.add_constant(X)).fit(q=tau).params for tau in taus}
print(pd.DataFrame(fits, index=["intercept", "slope"]).T)
```

Because the noise in this example does not depend on $x$, the fitted
slopes are similar at all quantiles and the lines are roughly parallel; a
spread of slopes across quantiles would indicate that the predictor affects
the shape of the conditional distribution, not only its location.

## Summary and Common Mistakes
### Summary

- `QuantileRegressor(quantile=tau, alpha=0.0)` and
  `sm.QuantReg(y, sm.add_constant(X)).fit(q=tau)` fit the same model.
- `QuantReg` reports standard errors and confidence intervals.
- Fitting several values of $\tau$ shows how the conditional distribution
  changes with the predictors.

### Common mistakes

- Leaving the default penalty `alpha=1.0` in `QuantileRegressor` when the
  unpenalized estimator is intended.
- Passing a column vector `y` of shape `(n, 1)` to `scikit-learn`; use a
  1-D array.
- Forgetting the intercept column for `QuantReg`.

## References

- scikit-learn developers. (n.d.). *scikit-learn: Machine learning in Python* [Software documentation]. <https://scikit-learn.org/>
- statsmodels developers. (n.d.). *statsmodels: Statistical models, hypothesis tests, and data exploration* [Software documentation]. <https://www.statsmodels.org/>

# Generalized Linear Models in Python

## Overview

This chapter fits the generalized linear models of Part IV in Python:
logistic regression for binary responses ({doc}`18-logistic-regression`) and
Poisson regression for counts ({doc}`19-poisson-and-count-regression`). It
uses `statsmodels`, which reports standard errors, tests, and residuals in
the style of statistical software, and `scikit-learn`, which focuses on
prediction, and it explains the differences between the two libraries'
defaults. It applies the models of {doc}`17-generalized-linear-models`, {doc}`18-logistic-regression`, and {doc}`19-poisson-and-count-regression`. The code was tested with `statsmodels` 0.14 and `scikit-learn` 1.9.

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- fit logistic regressions with `statsmodels` (`Logit`, `GLM`) and
  `scikit-learn` (`LogisticRegression`), and report odds ratios;
- implement the IRLS algorithm for logistic regression;
- fit Poisson regressions with offsets and exposure, and relate the
  attributes of a fitted model to the formulas of the theory chapter; and
- compute the Pearson dispersion statistic to check for overdispersion.
:::

## Logistic regression

### Example data

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(7)
n = 500
data_X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.binomial(1, 0.4, size=n)})
eta = -0.5 + 1.2 * data_X["x1"] - 0.8 * data_X["x2"]
data_y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
```

### Logistic regression with `statsmodels`

```python
from statsmodels.discrete.discrete_model import Logit

exog = sm.add_constant(data_X)
logit_res = Logit(data_y, exog).fit()
print(logit_res.summary())

odds_ratios = pd.DataFrame({
    "odds_ratio": np.exp(logit_res.params),
    "ci_lower": np.exp(logit_res.conf_int()[0]),
    "ci_upper": np.exp(logit_res.conf_int()[1]),
})
print(odds_ratios)
```

| Attribute | Quantity |
|---|---|
| `params`, `bse` | $\widehat{\bbeta}$ and Wald standard errors from $(\bX\T\widehat{\mathbf{W}}\bX)^{-1}$ |
| `tvalues`, `pvalues` | Wald statistics $z = \widehat{\beta}_j/\mathrm{SE}(\widehat{\beta}_j)$ and $p$-values (normal reference) |
| `llf`, `llnull` | $\ell(\widehat{\bbeta})$ and the intercept-only log-likelihood |
| `llr`, `llr_pvalue` | likelihood ratio statistic against the intercept-only model and its $\chi^2$ $p$-value |
| `prsquared` | McFadden's pseudo-$R^2 = 1 - \ell(\widehat{\bbeta})/\ell(\widehat{\beta}_0)$ |
| `predict()` | fitted probabilities $\widehat{\pi}_i$ |
| `get_margeff()` | average marginal effects $\beta_j\pi(1-\pi)$ |

The same model can be fitted as a GLM, which also reports the deviance;
for ungrouped binary data it equals $-2\ell(\widehat{\bbeta})$:

```python
glm_logit = sm.GLM(data_y, exog, family=sm.families.Binomial()).fit()
print(glm_logit.deviance, -2 * logit_res.llf)        # equal
```

### Logistic regression with `scikit-learn`

```python
from sklearn.linear_model import LogisticRegression

sk_logit = LogisticRegression(C=np.inf, max_iter=1000, tol=1e-10)   # no penalty
sk_logit.fit(data_X, data_y)
print(sk_logit.intercept_, sk_logit.coef_)
print(sk_logit.predict_proba(data_X)[:5, 1])          # fitted probabilities
```

`LogisticRegression` applies an $L_2$ penalty by default, controlled by the
inverse strength `C` (default `1.0`). Setting `C=np.inf` fits the
unpenalized maximum likelihood model. In versions of `scikit-learn` before
1.8 the same was done with `penalty=None`; that argument is deprecated from
version 1.8, and the former `multi_class` argument has been removed:
multinomial models are fitted automatically when the response has more than
two classes.

### IRLS from scratch

The Newton–Raphson (IRLS) iteration of {doc}`18-logistic-regression` takes a few
lines and reproduces `statsmodels` to machine precision:

```python
X = exog.to_numpy()
y = data_y
beta = np.zeros(X.shape[1])
for _ in range(25):
    pi = 1 / (1 + np.exp(-X @ beta))
    w = pi * (1 - pi)                                 # diagonal of W
    z = X @ beta + (y - pi) / w                       # working response
    beta = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * z))

se = np.sqrt(np.diag(np.linalg.inv(X.T @ (w[:, None] * X))))
print(np.max(np.abs(beta - logit_res.params.to_numpy())))   # about 1e-16
print(np.max(np.abs(se - logit_res.bse.to_numpy())))
```

## Poisson regression

### Example data

A simulated count data set with two predictors:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 500
data_X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.uniform(0, 2, size=n)})
true_beta = np.array([0.5, 0.3, -0.4])           # intercept, x1, x2
lam = np.exp(true_beta[0] + data_X.to_numpy() @ true_beta[1:])
data_Y = rng.poisson(lam)
```

### Poisson regression with `statsmodels`

#### Fitting the model

```python
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Poisson

exog = sm.add_constant(data_X)        # add the intercept column
model = Poisson(
    endog=data_Y, exog=exog,
    offset=None, exposure=None,
    missing="none", check_rank=True,
)
res = model.fit()
print(res.summary())

# the same model as a GLM with the Poisson family and log link
glm_pois = sm.GLM(data_Y, exog, family=sm.families.Poisson()).fit()
print(glm_pois.deviance, glm_pois.pearson_chi2)
```

The main parameters are:

`endog`, `exog`
: the response vector $\by$ (dependent, or endogenous, variable) and the
  predictor matrix $\bX$ (independent, or exogenous, variables). An
  intercept is **not** included by default; add it with
  `statsmodels.tools.add_constant` (available as `sm.add_constant`).

`offset`
: an offset $o_i$, added to the linear predictor with its coefficient fixed
  at one.

`exposure`
: an exposure $E_i$; $\ln(E_i)$ is added to the linear predictor with its
  coefficient fixed at one. Passing `exposure=E` is therefore equivalent to
  passing `offset=np.log(E)`.

#### Reading the results

For a fitted result `res`, the following attributes correspond to the
quantities in {doc}`19-poisson-and-count-regression`:

| Attribute | Quantity | Equivalent expression |
|---|---|---|
| `res.params` | $\widehat{\bbeta}$ | |
| `res.predict()` | fitted means $\widehat{\lambda}_i$ | `res.predict(model.exog)` (when there is no offset or exposure) |
| `res.fittedvalues` | linear predictor $\bx_i\T\widehat{\bbeta}$, excluding any offset | `np.log(res.predict())` (when there is no offset or exposure) |
| `res.resid` | raw residuals $y_i - \widehat{\lambda}_i$ | `model.endog - res.predict()` |
| `res.resid_pearson` | Pearson residuals $(y_i - \widehat{\lambda}_i)/\sqrt{\widehat{\lambda}_i}$ | |
| `res.df_resid` | residual degrees of freedom $n-p$ | |

Note that for discrete models such as `Poisson`, `fittedvalues` is the
**linear predictor**, not the fitted mean; use `predict()` for the fitted
counts.

#### Pearson dispersion

The dispersion estimate $\widehat{\phi} = \sum_i r_{P,i}^2/(n-p)$ from the
quasi-Poisson section of {doc}`19-poisson-and-count-regression` is computed
as follows:

```python
def compute_pearson_dispersion(res):
    """Pearson dispersion: sum of squared Pearson residuals / residual df."""
    return np.sum(np.square(res.resid_pearson)) / res.df_resid

print(compute_pearson_dispersion(res))   # close to 1 if there is no overdispersion
```

### Poisson regression with `scikit-learn`

```python
from sklearn.linear_model import PoissonRegressor

sk_pois = PoissonRegressor(alpha=0, fit_intercept=True, max_iter=1000)
sk_pois.fit(data_X, data_Y)

coef = pd.DataFrame(
    np.hstack([sk_pois.intercept_, sk_pois.coef_]),
    index=["const"] + sk_pois.feature_names_in_.tolist(),
    columns=["coef"],
)
print(coef)
```

`alpha=0` switches off the default $L_2$ penalty, so the estimates are
ordinary maximum likelihood estimates comparable with those of
`statsmodels`. The attribute `feature_names_in_` exists only when `data_X`
is a `pandas` DataFrame with column names. `scikit-learn` does not report
standard errors or test statistics.

### Why the two libraries can disagree

`Poisson` in `statsmodels` and `PoissonRegressor` in `scikit-learn` use
different optimization algorithms and different convergence criteria by
default, so their estimates may differ in the last few digits. Increasing
the number of iterations of `PoissonRegressor` (`max_iter`), or tightening
its tolerance (`tol`), usually makes the two sets of estimates agree.

## Summary and Common Mistakes
### Summary

- `Logit` or `GLM(..., family=Binomial())` in `statsmodels` gives estimates,
  Wald and likelihood ratio tests, and pseudo-$R^2$; exponentiate the
  coefficients and their confidence limits for odds ratios.
- `LogisticRegression(C=np.inf)` and `PoissonRegressor(alpha=0)` give the
  unpenalized maximum likelihood estimates in `scikit-learn`.
- `statsmodels.discrete.discrete_model.Poisson` handles offsets and
  exposure; `predict()` returns fitted means, `fittedvalues` the linear
  predictor, and `resid_pearson` the Pearson residuals.
- A short IRLS loop reproduces the library estimates.

### Common mistakes

- Forgetting `sm.add_constant`, which fits a model without an intercept.
- Leaving the default penalty in `LogisticRegression` (`C=1.0`) or
  `PoissonRegressor` (`alpha=1.0`) and then comparing the coefficients with
  unpenalized estimates.
- Reporting $e^{\beta_j}$ from a logistic regression as a ratio of
  probabilities; it is an odds ratio.
- Using `res.fittedvalues` as fitted counts; for discrete models it is the
  linear predictor.
- Passing a log-exposure as `exposure` (it would be logged twice); pass
  either `exposure=E` or `offset=np.log(E)`.

## References

- scikit-learn developers. (n.d.). *scikit-learn: Machine learning in Python* [Software documentation]. <https://scikit-learn.org/>
- statsmodels developers. (n.d.). *statsmodels: Statistical models, hypothesis tests, and data exploration* [Software documentation]. <https://www.statsmodels.org/>

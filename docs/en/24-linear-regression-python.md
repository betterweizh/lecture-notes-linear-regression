# Linear Regression in Python

## Overview

This chapter implements the linear regression methods of Parts II and III
in Python. It shows the `statsmodels` workflow, maps each attribute of a
fitted model to the formula it computes, gives a from-scratch implementation
of ordinary least squares that reproduces `statsmodels`, and provides code
for influence diagnostics and variance inflation factors. It applies the methods of {doc}`08-ols-and-ml-estimation`, {doc}`10-anova-and-inference`, {doc}`13-prediction-and-model-assessment`, and {doc}`14-standardization-and-multicollinearity`. The code was tested with Python 3.13, `numpy` 2.5, `pandas` 3.0, `scipy` 1.18, `statsmodels` 0.14, and `scikit-learn` 1.9.

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- fit a linear regression with `statsmodels` and read its estimates,
  tests, ANOVA quantities, and information criteria;
- compute confidence and prediction intervals for new observations;
- compute leverages, studentized residuals, and Cook's distances;
- compute variance inflation factors correctly, including the constant
  column; and
- implement OLS and its inference directly from the matrix formulas.
:::

## Example data

The examples use the diabetes data set shipped with `scikit-learn`: ten
baseline measurements (age, sex, body mass index, blood pressure, and six
blood serum measurements) of $n = 442$ patients, and a quantitative measure
of disease progression one year later as the response.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_diabetes

data_X, data_y = load_diabetes(return_X_y=True, as_frame=True, scaled=False)
X_train, y_train = data_X.iloc[:300], data_y.iloc[:300]
X_new = data_X.iloc[300:310]          # observations to predict
```

## The `statsmodels` workflow

Three objects are involved:

| Object | Role |
|---|---|
| [`statsmodels.regression.linear_model.OLS`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html) | the model: response and design matrix |
| [`RegressionResults`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html) | the fitted model, returned by `model.fit()` |
| [`PredictionResults`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.PredictionResults.html) | predictions, returned by `res.get_prediction()` |

```python
X = sm.add_constant(X_train)          # add the column of ones (intercept)
model = sm.OLS(y_train, X)
res = model.fit()
print(res.summary())

pred = res.get_prediction(sm.add_constant(X_new, has_constant="add"))
print(pred.summary_frame(alpha=0.05))
```

`OLS` does **not** add an intercept automatically; `sm.add_constant` adds a
column of ones named `const`. For a single new row, pass
`has_constant="add"`, because a one-row matrix would otherwise be mistaken
for a matrix that already contains a constant.

### Model information and fit

| Attribute | Meaning |
|---|---|
| `model.endog`, `model.exog` | response $\by$ (1-D array) and design matrix $\bX$ (2-D array) |
| `model.endog_names`, `model.exog_names` | variable names |
| `res.nobs` | number of observations $n$ |
| `res.summary(alpha=0.05)`, `res.summary2()` | summary tables |
| `res.fittedvalues` | fitted values $\widehat{\by}$ |
| `res.resid` | residuals $\be = \by - \widehat{\by}$ |
| `res.resid_pearson` | residuals divided by $\widehat{\sigma}$, i.e. the standardized residuals $d_i$ |

### Coefficients and $t$-tests

| Attribute | Formula |
|---|---|
| `res.params` | $\widehat{\bbeta} = (\bX\T\bX)^{-1}\bX\T\by$ |
| `res.scale` | $\widehat{\sigma}^2 = \mathit{SS}_E/(n-p)$ |
| `res.bse` | $\mathrm{SE}(\widehat{\beta}_j) = \sqrt{\widehat{\sigma}^2c_{jj}}$, $c_{jj}$ the $j$-th diagonal entry of $(\bX\T\bX)^{-1}$ |
| `res.tvalues`, `res.pvalues` | $T_j = \widehat{\beta}_j/\mathrm{SE}(\widehat{\beta}_j) \sim t(n-p)$ under $H_0:\beta_j = 0$, and two-sided $p$-values |
| `res.conf_int(alpha=0.05)` | $\widehat{\beta}_j \pm t_{1-\alpha/2,\,n-p}\,\mathrm{SE}(\widehat{\beta}_j)$ |
| `res.use_t` | whether inference uses the $t$ (rather than normal) distribution |

Here $p = k+1$ with an intercept and $p = k$ without one.

### ANOVA quantities

| Attribute | With intercept | Without intercept |
|---|---|---|
| `res.df_model` | $k$ | $k$ |
| `res.df_resid` | $n-k-1$ | $n-k$ |
| `res.ess` ($\mathit{SS}_R$) | $\sum_i(\widehat{y}_i-\bar{y})^2$ | $\sum_i\widehat{y}_i^2$ |
| `res.ssr` ($\mathit{SS}_E$) | $\sum_i(y_i-\widehat{y}_i)^2$ | $\sum_i(y_i-\widehat{y}_i)^2$ |
| `res.centered_tss`, `res.uncentered_tss` | $\sum_i(y_i-\bar{y})^2$ | $\sum_i y_i^2$ |
| `res.mse_model` ($\mathit{MS}_R$) | $\mathit{SS}_R/k$ | $\mathit{SS}_R/k$ |
| `res.mse_resid` ($\mathit{MS}_E$) | $\mathit{SS}_E/(n-k-1)$ | $\mathit{SS}_E/(n-k)$ |
| `res.mse_total` | $\mathit{SS}_T/(n-1)$ | $\mathit{SS}_T/n$ |
| `res.fvalue`, `res.f_pvalue` | $F = \mathit{MS}_R/\mathit{MS}_E \sim F(k, n-k-1)$ | $F \sim F(k, n-k)$ |

Note that `res.ssr` is the **residual** sum of squares $\mathit{SS}_E$, not
the regression sum of squares; `statsmodels` calls the regression sum of
squares `ess` ("explained sum of squares").

### Information criteria and goodness of fit

| Attribute | Formula |
|---|---|
| `res.llf` | maximized log-likelihood $\ln\widehat{\mathcal{L}} = -\frac{n}{2}\ln\bigl(2\pi\,\mathit{SS}_E/n\bigr) - \frac{n}{2}$ |
| `res.aic` | $2p - 2\ln\widehat{\mathcal{L}}$ |
| `res.bic` | $p\ln(n) - 2\ln\widehat{\mathcal{L}}$ |
| `res.rsquared` | $R^2 = 1 - \mathit{SS}_E/\mathit{SS}_T$ |
| `res.rsquared_adj` | $1 - \dfrac{\mathit{SS}_E/(n-p)}{\mathit{SS}_T/(n-1)}$ with an intercept; $1 - \dfrac{\mathit{SS}_E/(n-p)}{\mathit{SS}_T/n}$ without |

`statsmodels` counts the $p$ regression coefficients in AIC and BIC but not
$\sigma^2$, and it keeps the constant terms of the log-likelihood; as
explained in {doc}`15-variable-selection`, neither choice affects the
comparison of models fitted to the same data. The corrected AIC is not
provided and can be computed as
`res.aic + (2*p**2 + 2*p) / (res.nobs - p - 1)` with `p = len(res.params)`.

### Prediction

`res.get_prediction(X0)` returns a `PredictionResults` object. With
$\bx_0$ a row of `X0`:

| Attribute or method | Formula |
|---|---|
| `pred.predicted_mean` | $\widehat{y}_0 = \bx_0\T\widehat{\bbeta}$ |
| `pred.se_mean` | $\widehat{\sigma}\sqrt{\bx_0\T(\bX\T\bX)^{-1}\bx_0}$, for the confidence interval |
| `pred.se_obs` | $\widehat{\sigma}\sqrt{1 + \bx_0\T(\bX\T\bX)^{-1}\bx_0}$, for the prediction interval |
| `pred.conf_int(alpha=0.05)` | confidence interval for the mean response {eq}`eq-ci-mean` |
| `pred.conf_int(obs=True, alpha=0.05)` | prediction interval {eq}`eq-pi` |
| `pred.summary_frame(alpha=0.05)` | all of the above: `mean`, `mean_se`, `mean_ci_*`, `obs_ci_*` |

`res.predict(X0)` returns only the point predictions.

## Influence diagnostics

The leverages, studentized residuals, and Cook's distances of
{doc}`13-prediction-and-model-assessment` are available from
`res.get_influence()`:

```python
infl = res.get_influence()
diag = pd.DataFrame({
    "leverage": infl.hat_matrix_diag,                 # h_ii
    "student_resid": infl.resid_studentized_internal, # r_i
    "cooks_d": infl.cooks_distance[0],                # D_i
}, index=X_train.index)
print(diag.sort_values("cooks_d", ascending=False).head())
```

## Variance inflation factors

`statsmodels.stats.outliers_influence.variance_inflation_factor(exog, j)`
regresses column `j` of `exog` on the **other columns of `exog`** and
returns $1/(1-R_j^2)$. It does not add an intercept, so the matrix passed in
must contain a constant column; otherwise $R_j^2$ is computed without an
intercept and the VIFs are far too large. For the diabetes data, omitting
the constant gives a VIF of about 55 for `bmi` instead of the correct 1.5.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(data):
    """VIF of each column of a DataFrame of predictors (no constant column)."""
    exog = sm.add_constant(data)
    return pd.Series(
        {col: variance_inflation_factor(exog.values, i)
         for i, col in enumerate(exog.columns) if col != "const"})

def stepwise_remove_variable_vif(data, threshold=10.0, verbose=True):
    """Repeatedly drop the predictor with the largest VIF until all VIFs < threshold."""
    removed, history, step = [], {}, 0
    while data.shape[1] > 1:
        vif = calculate_vif(data)
        history[f"step{step}"] = vif
        if (vif < threshold).all():
            break
        worst = vif.idxmax()
        if verbose:
            print(f"Step {step}: remove {worst} (VIF = {vif.max():.3f})")
        removed.append(worst)
        data = data.drop(columns=worst)
        step += 1
    return data, removed, pd.concat(history, axis=1)

print(calculate_vif(data_X).round(2))
kept, removed, history = stepwise_remove_variable_vif(data_X, threshold=10)
```

For the diabetes data, the serum measurements `s1` and `s2` have VIFs of
about 59 and 39; the stepwise procedure removes `s1` first. As discussed in
{doc}`14-standardization-and-multicollinearity`, removing a predictor is only
one of several remedies, and subject-matter considerations should guide the
choice.

## OLS from scratch

The class below computes every quantity in the tables above directly from
the formulas of {doc}`08-ols-and-ml-estimation`, {doc}`10-anova-and-inference`,
and {doc}`13-prediction-and-model-assessment`. Its results agree with
`statsmodels` to within about $10^{-9}$ (relative), with and without an
intercept.

:::{dropdown} The `LinearRegression` class
```python
import numpy as np
import pandas as pd
from scipy import stats

class LinearRegression:
    """Ordinary least squares from the matrix formulas of Part II.

    Parameters
    ----------
    data_X : pandas.DataFrame or numpy.ndarray, shape (n, k)
        Predictors, without a column of ones.
    data_y : pandas.Series or numpy.ndarray, shape (n,)
        Response.
    intercept : bool
        Whether to add an intercept (a column of ones).
    """

    def __init__(self, data_X, data_y, intercept=True):
        X = np.asarray(data_X, dtype=float)
        y = np.asarray(data_y, dtype=float).ravel()
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError(f"incompatible shapes: X {X.shape}, y {y.shape}")

        self.intercept = intercept
        self.nobs, self.n_indep_var = X.shape            # n and k
        self.exog_names = (list(data_X.columns) if isinstance(data_X, pd.DataFrame)
                           else [f"x{j + 1}" for j in range(self.n_indep_var)])
        self.endog_names = getattr(data_y, "name", None) or "y"

        if intercept:
            X = np.column_stack([np.ones(self.nobs), X])
            self.exog_names = ["const"] + self.exog_names
        self.exog, self.endog = X, y
        self.num_params = X.shape[1]                      # p

        self.df_model = self.n_indep_var                  # k
        self.df_resid = self.nobs - self.num_params       # n - p
        self.df_total = self.df_model + self.df_resid     # n - 1 (or n)

    # ------------------------------------------------------------------
    def fit(self):
        X, y = self.exog, self.endog
        self.XTX_inv = np.linalg.inv(X.T @ X)             # C = (X'X)^{-1}
        self.params = self.XTX_inv @ X.T @ y              # beta_hat
        self.hat_diag = np.einsum("ij,jk,ik->i", X, self.XTX_inv, X)  # h_ii
        self.fittedvalues = X @ self.params
        self.resid = y - self.fittedvalues

        # sums of squares (centered total SS with an intercept, uncentered without)
        self.ssr = float(self.resid @ self.resid)         # SS_E
        if self.intercept:
            self.tss = float(np.sum((y - y.mean()) ** 2))
        else:
            self.tss = float(y @ y)
        self.ess = self.tss - self.ssr                    # SS_R

        self.scale = self.ssr / self.df_resid             # sigma^2 hat = MS_E
        self.resid_pearson = self.resid / np.sqrt(self.scale)
        self.bse = np.sqrt(self.scale * np.diag(self.XTX_inv))
        return self

    # ------------------------------------------------------------------
    def anova(self):
        ms_r = self.ess / self.df_model
        ms_e = self.ssr / self.df_resid
        f = ms_r / ms_e
        table = pd.DataFrame(
            {"df": [self.df_model, self.df_resid, self.df_total],
             "SS": [self.ess, self.ssr, self.tss],
             "MS": [ms_r, ms_e, self.tss / self.df_total],
             "F": [f, np.nan, np.nan],
             "P-value": [stats.f.sf(f, self.df_model, self.df_resid), np.nan, np.nan]},
            index=["Regression", "Residual", "Total"])
        return table

    def coef_table(self, alpha=0.05):
        t = self.params / self.bse
        q = stats.t.ppf(1 - alpha / 2, self.df_resid)
        return pd.DataFrame(
            {"coef": self.params, "std_err": self.bse, "t": t,
             "p_value": 2 * stats.t.sf(np.abs(t), self.df_resid),
             "ci_lower": self.params - q * self.bse,
             "ci_upper": self.params + q * self.bse},
            index=self.exog_names)

    def criteria(self):
        n, p = self.nobs, self.num_params
        llf = -n / 2 * (np.log(2 * np.pi * self.ssr / n) + 1)   # max log-likelihood
        aic = 2 * p - 2 * llf             # statsmodels convention: sigma^2 not counted
        return pd.Series({
            "llf": llf,
            "aic": aic,
            "aicc": aic + (2 * p ** 2 + 2 * p) / (n - p - 1),
            "bic": p * np.log(n) - 2 * llf,
            "rsquared": 1 - self.ssr / self.tss,
            "rsquared_adj": 1 - (self.ssr / self.df_resid) / (self.tss / self.df_total),
        })

    def influence(self):
        h = self.hat_diag
        r = self.resid / np.sqrt(self.scale * (1 - h))   # studentized residuals
        cooks = r ** 2 / self.num_params * h / (1 - h)
        return pd.DataFrame({"leverage": h, "student_resid": r, "cooks_d": cooks})

    def predict(self, data_X_new, alpha=0.05):
        X0 = np.asarray(data_X_new, dtype=float)
        if self.intercept:
            X0 = np.column_stack([np.ones(X0.shape[0]), X0])
        mean = X0 @ self.params
        q = np.einsum("ij,jk,ik->i", X0, self.XTX_inv, X0)   # x0' C x0
        se_mean = np.sqrt(self.scale * q)
        se_obs = np.sqrt(self.scale * (1 + q))
        t = stats.t.ppf(1 - alpha / 2, self.df_resid)
        return pd.DataFrame({
            "mean": mean, "mean_se": se_mean, "obs_se": se_obs,
            "mean_ci_lower": mean - t * se_mean, "mean_ci_upper": mean + t * se_mean,
            "obs_ci_lower": mean - t * se_obs, "obs_ci_upper": mean + t * se_obs})
```
:::

The following code fits both implementations and reports the largest
relative difference:

```python
from scipy import stats   # used inside the class

for intercept in (True, False):
    mine = LinearRegression(X_train, y_train, intercept=intercept).fit()
    exog = sm.add_constant(X_train) if intercept else X_train
    ref = sm.OLS(y_train, exog).fit()
    diffs = {
        "params": np.max(np.abs(mine.params - ref.params) / np.abs(ref.params)),
        "bse": np.max(np.abs(mine.bse - ref.bse) / ref.bse),
        "F": abs(mine.anova().loc["Regression", "F"] - ref.fvalue) / ref.fvalue,
        "aic": abs(mine.criteria()["aic"] - ref.aic) / abs(ref.aic),
    }
    print(intercept, {k: f"{v:.1e}" for k, v in diffs.items()})
```

Two implementation notes:

- `np.linalg.inv(X.T @ X)` is used for clarity because it mirrors the
  formulas. For badly conditioned design matrices, a QR decomposition
  (`np.linalg.lstsq`) is numerically more stable; `statsmodels` uses a
  pseudo-inverse.
- Without an intercept, the total sum of squares is uncentered
  ($\sum_i y_i^2$), as in {doc}`10-anova-and-inference`.

## Summary and Common Mistakes
### Summary

- `sm.OLS(y, sm.add_constant(X)).fit()` returns estimates, $t$- and
  $F$-tests, ANOVA sums of squares, and AIC/BIC; `get_prediction` gives
  confidence and prediction intervals.
- `get_influence()` gives leverages, studentized residuals, and Cook's
  distances.
- VIFs require a constant column in the matrix passed to
  `variance_inflation_factor`.
- The matrix formulas of Part II reproduce `statsmodels` to numerical
  precision.

### Common mistakes

- Forgetting `sm.add_constant`, which silently fits a model through the
  origin.
- Reading `res.ssr` as the regression sum of squares; it is $\mathit{SS}_E$.
- Computing VIFs without a constant column.
- Using `pred.conf_int()` when a prediction interval is needed; pass
  `obs=True`.

## References

- scikit-learn developers. (n.d.). *scikit-learn: Machine learning in Python* [Software documentation]. <https://scikit-learn.org/>
- statsmodels developers. (n.d.). *statsmodels: Statistical models, hypothesis tests, and data exploration* [Software documentation]. <https://www.statsmodels.org/>

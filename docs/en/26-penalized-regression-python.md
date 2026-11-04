# Ridge and Lasso Regression in Python

## Overview

This chapter fits the penalized regressions of {doc}`16-ridge-and-lasso` with
`scikit-learn`. It explains how the library scales the objective, so that
its `alpha` parameter can be related to the penalty $\lambda$ of the theory
chapter, shows how to standardize the predictors inside a pipeline, chooses
the penalty by cross-validation, and verifies the closed-form ridge
estimator and the coordinate descent algorithm for the lasso. It applies the methods of {doc}`16-ridge-and-lasso` and {doc}`24-linear-regression-python`. The code was tested with `scikit-learn` 1.9.

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- fit ridge, lasso, and elastic net regressions with standardized
  predictors;
- convert between the `alpha` of `scikit-learn` and the $\lambda$ of
  {doc}`16-ridge-and-lasso`;
- choose the penalty by $K$-fold cross-validation and compute a
  regularization path; and
- implement the ridge closed form and lasso coordinate descent and check
  them against the library.
:::

## Objective functions in `scikit-learn`

`scikit-learn` names the penalty parameter `alpha` and scales the objectives
differently for different estimators:

| Estimator | Objective minimized | Relation to {eq}`eq-penalized-ls` |
|---|---|---|
| `Ridge(alpha)` | $\lVert\by - \bX\bbeta\rVert_2^2 + \alpha\lVert\bbeta\rVert_2^2$ | $\lambda = \alpha$ |
| `Lasso(alpha)` | $\dfrac{1}{2n}\lVert\by - \bX\bbeta\rVert_2^2 + \alpha\lVert\bbeta\rVert_1$ | $\lambda = 2n\alpha$ |
| `ElasticNet(alpha, l1_ratio)` | $\dfrac{1}{2n}\lVert\by - \bX\bbeta\rVert_2^2 + \alpha\rho\lVert\bbeta\rVert_1 + \dfrac{\alpha(1-\rho)}{2}\lVert\bbeta\rVert_2^2$, $\rho$ = `l1_ratio` | $\lambda_1 = 2n\alpha\rho$, $\lambda_2 = n\alpha(1-\rho)$ |

All three fit an unpenalized intercept by default (`fit_intercept=True`).
None of them standardizes the predictors, so standardization must be done
explicitly.

## Example data

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, lasso_path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True, as_frame=True, scaled=False)
n, k = X.shape
```

## Ridge regression

Standardizing inside a pipeline ensures that the same centering and scaling
are applied to new data at prediction time, and, during cross-validation,
that they are estimated from the training folds only.

```python
ridge = make_pipeline(StandardScaler(), Ridge(alpha=25.0))
ridge.fit(X, y)
coef_std = ridge[-1].coef_                       # coefficients on the standardized scale
coef_orig = coef_std / ridge[0].scale_           # back to the original units
print(pd.Series(coef_orig, index=X.columns).round(3))
```

On the standardized scale, the coefficient of predictor $j$ is the change in
$y$ for a one-standard-deviation change in $x_j$; dividing by the standard
deviation (`scale_`) converts it to the original units, as in
{doc}`14-standardization-and-multicollinearity`. Note that `StandardScaler`
divides by the standard deviation computed with denominator $n$.

### Checking the closed form

The ridge estimator {eq}`eq-ridge` can be computed directly on the
standardized predictors and the centered response:

```python
Z = StandardScaler().fit_transform(X)
yc = y - y.mean()
lam = 25.0
beta_closed = np.linalg.solve(Z.T @ Z + lam * np.eye(k), Z.T @ yc)
beta_sklearn = Ridge(alpha=lam).fit(Z, y).coef_
print(np.max(np.abs(beta_closed - beta_sklearn)))     # about 1e-14
```

### Effective degrees of freedom

```python
d = np.linalg.svd(Z, compute_uv=False)                 # singular values d_j
df_ridge = np.sum(d**2 / (d**2 + lam))                 # df(lambda)
H = Z @ np.linalg.solve(Z.T @ Z + lam * np.eye(k), Z.T)
print(df_ridge, np.trace(H))                           # equal
```

## Lasso regression

```python
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.5))
lasso.fit(X, y)
coef = pd.Series(lasso[-1].coef_, index=X.columns)
print(coef.round(3))
print("number of zero coefficients:", int((coef == 0).sum()))
```

With `alpha = 0.5`, two of the ten coefficients are exactly zero; larger
values of `alpha` set more coefficients to zero.

### Coordinate descent from scratch

The following function implements {prf:ref}`alg-lasso-cd` with the
objective $\lVert\by-\bX\bbeta\rVert_2^2 + \lambda\lVert\bbeta\rVert_1$. With
$\lambda = 2n\alpha$ it reproduces `Lasso(alpha)`:

```python
def soft_threshold(z, gamma):
    """S(z, gamma) = sign(z) * max(|z| - gamma, 0)."""
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)

def lasso_cd(Z, y, lam, tol=1e-12, max_iter=100_000):
    """Lasso by cyclic coordinate descent; Z standardized, y centered."""
    beta = np.zeros(Z.shape[1])
    col_ss = np.sum(Z**2, axis=0)                      # x_j' x_j
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(Z.shape[1]):
            r_j = y - Z @ beta + Z[:, j] * beta[j]     # partial residual
            beta[j] = soft_threshold(Z[:, j] @ r_j, lam / 2) / col_ss[j]
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta

alpha = 0.5
beta_cd = lasso_cd(Z, yc, lam=2 * n * alpha)
beta_skl = Lasso(alpha=alpha, tol=1e-12, max_iter=100_000).fit(Z, y).coef_
print(np.max(np.abs(beta_cd - beta_skl)))             # about 1e-10

lam_max = 2 * np.max(np.abs(Z.T @ yc))                 # all coefficients zero beyond this
print(np.all(lasso_cd(Z, yc, 1.0001 * lam_max) == 0))  # True
```

This implementation recomputes the full residual for clarity; production
code updates the residual incrementally.

## Choosing the penalty by cross-validation

```python
alphas = np.logspace(-3, 3, 100)

ridge_cv = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
ridge_cv.fit(X, y)
print("ridge alpha:", ridge_cv[-1].alpha_)

lasso_cv = make_pipeline(StandardScaler(), LassoCV(cv=10, random_state=0))
lasso_cv.fit(X, y)
print("lasso alpha:", lasso_cv[-1].alpha_)
print(pd.Series(lasso_cv[-1].coef_, index=X.columns).round(3))
```

By default `RidgeCV` uses the efficient leave-one-out formula of
{doc}`16-ridge-and-lasso`, while `LassoCV` uses $K$-fold cross-validation over
an automatically chosen grid. `LassoCV` stores the cross-validation errors
in `mse_path_` (one column per fold), from which the one-standard-error rule
can be applied:

```python
cv = lasso_cv[-1]
mean_mse = cv.mse_path_.mean(axis=1)
se_mse = cv.mse_path_.std(axis=1, ddof=1) / np.sqrt(cv.mse_path_.shape[1])
i_min = np.argmin(mean_mse)
alpha_1se = np.max(cv.alphas_[mean_mse <= mean_mse[i_min] + se_mse[i_min]])
print("alpha (min):", cv.alphas_[i_min], " alpha (1-SE):", alpha_1se)
```

## Regularization path

```python
import matplotlib.pyplot as plt

alphas_path, coefs_path, _ = lasso_path(Z, yc, alphas=np.logspace(-2, 2, 100))
fig, ax = plt.subplots(figsize=(7, 4))
for j, name in enumerate(X.columns):
    ax.plot(np.log10(alphas_path), coefs_path[j], label=name)
ax.set_xlabel(r"$\log_{10}(\alpha)$")
ax.set_ylabel("standardized coefficient")
ax.set_title("Lasso regularization path")
ax.legend(ncol=2, fontsize=8)
plt.show()
```

Reading the plot from right to left, coefficients enter the model one at a
time as the penalty decreases; at the far left they approach the OLS
estimates.

## Summary and Common Mistakes
### Summary

- `Ridge(alpha)` uses $\lambda = \alpha$; `Lasso(alpha)` uses
  $\lambda = 2n\alpha$.
- Put `StandardScaler` and the estimator in one pipeline.
- `RidgeCV` and `LassoCV` choose the penalty by cross-validation;
  `lasso_path` computes the regularization path.
- The closed-form ridge estimator and a short coordinate descent loop
  reproduce `scikit-learn` to numerical precision.

### Common mistakes

- Fitting `Ridge` or `Lasso` on unstandardized predictors.
- Standardizing the whole data set before cross-validation, which lets
  information from the validation folds leak into training; standardize
  inside a pipeline.
- Using the same `alpha` for `Ridge` and `Lasso` and expecting comparable
  amounts of shrinkage; the objectives are scaled differently.
- Interpreting standardized coefficients in original units without
  dividing by `scale_`.

## References

- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software, 33*(1), 1–22. <https://doi.org/10.18637/jss.v033.i01>
- scikit-learn developers. (n.d.). *scikit-learn: Machine learning in Python* [Software documentation]. <https://scikit-learn.org/>

# Standardization and Multicollinearity

## Overview

Regression coefficients are measured in the units of their predictors, so
their magnitudes cannot be compared directly. Standardizing the variables
puts all coefficients on a common scale. The same standardized form also
gives the cleanest description of **multicollinearity**: strong linear
relationships among predictors that inflate the variances of the
coefficient estimators. It builds on the covariance of $\widehat{\bbeta}$ ({doc}`09-ols-properties-and-geometry`) and on $t$- and $F$-tests ({doc}`10-anova-and-inference`).

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- fit a standardized regression and convert its coefficients back to the
  original scale;
- explain how multicollinearity affects estimates, standard errors, and
  tests; and
- compute and interpret variance inflation factors (VIFs).
:::

## Standardized linear regression

### Standardizing the variables

Center each variable at its sample mean and divide by its sample standard
deviation:

$$
\begin{gathered}
y_i^{*} = \frac{y_i - \bar{y}}{s_y}, \\
x_{ij}^{*} = \frac{x_{ij} - \bar{x}_j}{s_{x_j}}, \qquad i = 1,\ldots,n,\ \ j = 1,\ldots,k,
\end{gathered}
$$

where $s_y$ and $s_{x_j}$ are the sample standard deviations of the
response $y$ and of predictor $x_j$. Each standardized variable has mean
zero and variance one.

### The standardized model

Regressing the standardized response on the standardized predictors gives

$$
y_i^{*} = \beta_1^{*}x_{i1}^{*} + \beta_2^{*}x_{i2}^{*} + \cdots + \beta_k^{*}x_{ik}^{*} + \varepsilon_i^{*},
\qquad i = 1,\ldots,n .
$$

The model has no intercept, because all standardized variables have mean
zero and the fitted plane passes through the origin. The coefficient
$\beta_j^{*}$ is the expected change in $y$, measured in standard deviations
of $y$, for a one-standard-deviation increase in $x_j$ with the other
predictors held fixed. Because the coefficients are unit-free, they can be
compared across predictors.

### Converting between scales

The OLS estimates on the two scales are related by

$$
\begin{gathered}
\widehat{\beta}_j^{*} = \widehat{\beta}_j\,\frac{s_{x_j}}{s_y}, \\
\widehat{\beta}_j = \widehat{\beta}_j^{*}\,\frac{s_y}{s_{x_j}}, \qquad j = 1,\ldots,k,
\end{gathered}
$$

and the intercept of the original model is recovered as
$\widehat{\beta}_0 = \bar{y} - \sum_{j=1}^k \widehat{\beta}_j\bar{x}_j$.
Standardization is a linear change of units, so the back-transformed
fitted values and $R^2$ are unchanged.

### Correlation form

Let $\mathbf{Z}$ be the $n\times k$ matrix with entries
$z_{ij} = (x_{ij}-\bar{x}_j)/\sqrt{S_{jj}}$, where
$S_{jj} = \sum_{i=1}^n (x_{ij}-\bar{x}_j)^2$. Then
$\mathbf{R}_{xx} = \mathbf{Z}\T\mathbf{Z}$ is the $k\times k$ **correlation
matrix** of the predictors. This "unit-length" scaling differs from the one
above only by the constant factor $\sqrt{n-1}$; if the response is scaled
in the same way, it gives the same standardized coefficients. It is
convenient for studying multicollinearity.

## Multicollinearity

### What goes wrong

Ordinary least squares does not require the predictors to be uncorrelated,
only that $\bX$ has full column rank ({prf:ref}`lem-gram-invertible`). When
some predictors are strongly correlated, however, we say that
**multicollinearity** is present, and two problems arise.

1. **Inflated variances.** Multicollinearity reduces the precision of the
   coefficient estimates: the standard errors $\mathrm{SE}(\widehat{\beta}_j)$
   grow and the $t$ statistics shrink. We may then fail to reject
   $H_0:\beta_j = 0$ even though $\beta_j \ne 0$; the **power** of the test,
   the probability of detecting a nonzero coefficient, is reduced.
2. **Numerical instability.** When the columns of $\bX$ are nearly linearly
   dependent, $\bX\T\bX$ is nearly singular (ill-conditioned) and its inverse
   is numerically unstable. Under exact collinearity $\bX\T\bX$ is not
   invertible and $\widehat{\bbeta}$ is not unique.

Pairwise correlation coefficients detect linear relationships between two
predictors, but they cannot reveal a near-linear dependence among three or
more predictors. The variance inflation factor addresses this.

### Variance inflation factor

Let $R_j^2$ be the coefficient of determination from regressing $x_j$, as
the response, on the other $k-1$ predictors (with an intercept):

$$
x_{ij} = \gamma_0 + \sum_{l \ne j}\gamma_l x_{il} + u_i .
$$

The variance of the slope estimator in the original model can be written as

$$
\Var(\widehat{\beta}_j) = \frac{\sigma^2}{S_{jj}}\cdot\frac{1}{1-R_j^2},
\qquad j = 1,\ldots,k .
$$

In correlation form, the $j$-th diagonal entry of $\mathbf{R}_{xx}^{-1}$ is
exactly $1/(1-R_j^2)$. The factor

$$
\mathrm{VIF}_j = \frac{1}{1-R_j^2}, \qquad j = 1,\ldots,k,
$$ (eq-vif)

is the **variance inflation factor**: it is the factor by which
$\Var(\widehat{\beta}_j)$ is inflated relative to the ideal case in which
$x_j$ is uncorrelated with the other predictors ($R_j^2 = 0$). The stronger
the linear dependence of $x_j$ on the others, the closer $R_j^2$ is to one
and the larger the VIF.

Commonly used thresholds:

- a VIF above $5$ or $10$ (i.e. $R_j^2 > 0.8$ or $0.9$) indicates a
  problematic amount of collinearity (James et al., 2013, p. 101);
- Montgomery and Runger (2018, p. 350) suggest that VIFs should not exceed
  $10$, or, by stricter standards, $4$ or $5$.

When a VIF is large, possible remedies include removing or combining
redundant predictors, collecting more data, or using a penalized method such
as ridge regression ({doc}`16-ridge-and-lasso`). Code for computing VIFs is
given in {doc}`24-linear-regression-python` and {doc}`25-linear-regression-r`.

### Remarks

- Multicollinearity mainly affects the individual coefficients. Even when
  some $\widehat{\beta}_j$ have large VIFs, the fitted values $\widehat{y}$
  and predictions inside the range of the data may still be accurate.
- A typical symptom is a significant overall $F$-test combined with
  non-significant $t$-tests for several individual coefficients.

## Summary and Common Mistakes
### Summary

- Standardized coefficients $\beta_j^{*} = \beta_j s_{x_j}/s_y$ are
  unit-free and comparable across predictors.
- Multicollinearity inflates $\Var(\widehat{\beta}_j)$ by
  $\mathrm{VIF}_j = 1/(1-R_j^2)$.
- VIFs above about $5$–$10$ indicate harmful collinearity.

### Common mistakes

- Comparing raw coefficients of predictors measured in different units to
  judge their importance.
- Forgetting the factor $s_{x_j}/s_y$ (not $s_{x_1}/s_y$) when converting a
  coefficient to the standardized scale.
- Reading pairwise correlations as a complete check for multicollinearity.

## References

- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An introduction to statistical learning: With applications in R*. Springer. <https://doi.org/10.1007/978-1-4614-7138-7>
- Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley.

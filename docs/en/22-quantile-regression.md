# Quantile Regression

## Overview

All earlier chapters model a conditional **mean**: $\E[Y\mid\bx]$ in linear
regression and $g^{-1}(\bx\T\bbeta)$ in GLMs. The mean summarizes only the
center of the conditional distribution. **Quantile regression** models
conditional **quantiles** instead, for example the median or the 10th
percentile, so that different parts of the response distribution can depend
on the predictors in different ways. It replaces the squared-error loss of
least squares with the asymmetric pinball loss. It builds on least squares viewed as loss minimization ({doc}`08-ols-and-ml-estimation`).

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- define the $\tau$-quantile of a random variable;
- write the pinball (quantile) loss and explain its asymmetry;
- formulate the linear quantile regression model and its estimation
  problem; and
- contrast quantile regression with conditional-mean regression.
:::

## Quantiles

For a random variable $X$ with cumulative distribution function $F$, the
**$\tau$-quantile**, $0 < \tau < 1$, is the value $x_\tau$ such that

$$
\begin{aligned}
\tau &= \Prob(X \le x_\tau) \\
&= F(x_\tau).
\end{aligned}
$$

For distributions with jumps or flat parts, the general definition is
$x_\tau = \inf\{x : F(x) \ge \tau\}$. The median is the $0.5$-quantile.

## The pinball loss

Least squares minimizes the mean squared error, the $L_2$ loss. Quantile
regression instead uses the **pinball loss** (also called the quantile loss
or check function) $\rho_\tau(u)$, applied to the residual $u = y - \widehat{y}$:

$$
\rho_\tau(u) =
\begin{cases}
\tau\,u, & u \ge 0, \\
(\tau - 1)\,u, & u < 0,
\end{cases}
$$ (eq-pinball)

or, equivalently,

$$
\rho_\tau(u) = \tau\max(u,0) + (1-\tau)\max(-u,0).
$$

Both branches are non-negative: a positive residual (under-prediction) is
weighted by $\tau$, and a negative residual (over-prediction) by $1-\tau$.
Minimizing $\E[\rho_\tau(Y - q)]$ over $q$ gives the $\tau$-quantile of $Y$.

## The quantile regression model

For a given $\tau\in(0,1)$, linear quantile regression models the
conditional $\tau$-quantile as $Q_\tau(Y\mid\bx) = \bx\T\bbeta_\tau$ and
estimates the coefficient vector by

$$
\begin{aligned}
\widehat{\bbeta}_\tau
&= \operatorname*{arg\,min}_{\bbeta_\tau}\ \sum_{i=1}^n \rho_\tau\bigl(y_i - \bx_i\T\bbeta_\tau\bigr) \\
&= \operatorname*{arg\,min}_{\bbeta_\tau}\ \Biggl[\ \sum_{i:\ y_i \ge \bx_i\T\bbeta_\tau}\tau\bigl(y_i - \bx_i\T\bbeta_\tau\bigr)
 + \sum_{i:\ y_i < \bx_i\T\bbeta_\tau}(\tau-1)\bigl(y_i - \bx_i\T\bbeta_\tau\bigr)\Biggr].
\end{aligned}
$$ (eq-quantile-regression)

Each $\tau$ has its own coefficient vector $\bbeta_\tau$.

- With $\tau = 0.5$, $\rho_{0.5}(u) = |u|/2$, so the problem is **median
  regression**, also called least absolute deviations: it minimizes the mean
  absolute error, the $L_1$ loss.
- Higher quantiles penalize under-prediction more heavily and lower
  quantiles penalize over-prediction more heavily. For the $0.25$-quantile,
  over-predicting $y$ has weight $0.75$ and under-predicting has weight
  $0.25$, so the fitted line lies low; for the $0.75$-quantile the weights
  are reversed and the fitted line lies high.

### Conditional mean versus conditional quantiles

Least squares estimates how the **center** of the response distribution
depends on $\bx$. Quantile regression, fitted at several values of $\tau$,
shows how the whole distribution shifts, spreads, or becomes skewed with
$\bx$. It is useful when the spread of the response changes with the
predictors (heteroskedasticity), when interest lies in the tails (for
example, the 90th percentile of travel time), or when outliers in $y$ would
distort a mean regression; median regression is more robust to such
outliers than least squares.

### Estimation

The objective {eq}`eq-quantile-regression` is piecewise linear and not
differentiable at zero residuals, so there is no closed-form solution like
{eq}`eq-ols`. The problem can be written as a linear program. The main
estimation algorithms are the simplex method, interior-point methods, and
smoothing algorithms.

## Implementation

Fitting quantile regressions with `scikit-learn` and `statsmodels` is shown
in {doc}`28-quantile-regression-python`.

## Summary and Common Mistakes
### Summary

- The $\tau$-quantile solves $\min_q \E[\rho_\tau(Y-q)]$ with the pinball
  loss $\rho_\tau$.
- Linear quantile regression models $Q_\tau(Y\mid\bx) = \bx\T\bbeta_\tau$
  and estimates $\bbeta_\tau$ by minimizing the total pinball loss.
- $\tau = 0.5$ gives median ($L_1$) regression.
- Fitting several quantiles describes how the whole conditional
  distribution depends on the predictors.

### Common mistakes

- Interpreting $\bbeta_\tau$ as the effect on the mean; it is the effect on
  the conditional $\tau$-quantile.
- Expecting quantile lines fitted separately at different $\tau$ never to
  cross; with finite samples they occasionally do.

## References

- Lianxh [连享会]. (n.d.). *Stata+R：分位数回归一文读懂* [Quantile regression with Stata and R]. <https://www.lianxh.cn/news/f29d83cd303dc.html>

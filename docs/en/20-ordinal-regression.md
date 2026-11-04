# Ordinal Regression

## Overview

An **ordinal** response has categories with a natural order but no
meaningful distance between them, for example *unlikely* < *somewhat
likely* < *very likely*. Treating such a response as numeric ignores that
the gaps between categories are unknown; treating it as unordered (as in
{doc}`21-multinomial-choice-models`) ignores the order. **Cumulative link
models** use the order directly by modeling the probability of falling at
or below each category. They extend binary logit and probit regression and
fit within the GLM framework of {doc}`17-generalized-linear-models`. It builds on link functions ({doc}`17-generalized-linear-models`) and on binary logit and probit models ({doc}`18-logistic-regression`).

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- define cumulative probabilities, cumulative odds, and cut-points for an
  ordinal response;
- write the ordinal logit (proportional odds) and ordinal probit models;
- interpret the coefficients of the ordinal logit model as cumulative odds
  ratios; and
- write the log-likelihood used to estimate the model.
:::

## Setting

Let $Y$ be an ordinal response with $J$ ordered categories
$1 < 2 < \cdots < J$, and let $\bx$ be a vector of predictors **without** an
intercept column (the cut-points below play the role of intercepts). The
**cumulative probability** of category $j$ is

$$
\Prob(Y \le j \mid \bx), \qquad j = 1,\ldots,J-1,
$$

with $\Prob(Y \le J\mid\bx) = 1$. The **cumulative odds** of being in
category $j$ or below are

$$
\frac{\Prob(Y \le j\mid\bx)}{\Prob(Y > j\mid\bx)}, \qquad j = 1,\ldots,J-1 .
$$

## Cumulative link models

A cumulative link model specifies

$$
F^{-1}\bigl(\Prob(Y \le j\mid\bx)\bigr) = \theta_j - \bbeta\T\bx,
\qquad j = 1,\ldots,J-1,
$$ (eq-cumulative-link)

where $F$ is a continuous cumulative distribution function (CDF) and the
**cut-points** (thresholds) satisfy
$\theta_1 < \theta_2 < \cdots < \theta_{J-1}$. The same coefficient vector
$\bbeta$ is used for every $j$; only the cut-points change. The minus sign
is a common convention: a positive $\beta_l$ shifts probability toward the
higher categories.

### Ordinal logit model

The log of the cumulative odds is the **logit** of the cumulative
probability,

$$
\ln\!\left(\frac{\Prob(Y \le j\mid\bx)}{\Prob(Y > j\mid\bx)}\right)
= \logit\bigl(\Prob(Y \le j\mid\bx)\bigr).
$$

The **ordinal logistic regression** model takes $F$ to be the logistic CDF:

$$
\logit\bigl(\Prob(Y \le j\mid\bx)\bigr) = \theta_j - \bbeta\T\bx,
\qquad j = 1,\ldots,J-1 .
$$ (eq-ordinal-logit)

For two predictor vectors $\bx_a$ and $\bx_b$, the difference of the
cumulative log odds is $-\bbeta\T(\bx_a - \bx_b)$ for every $j$. The
cumulative odds ratio therefore does not depend on $j$, which is why the
model is also called the **proportional odds model**: a one-unit increase in
$x_l$ multiplies the odds of $Y \le j$ by $e^{-\beta_l}$ (equivalently,
multiplies the odds of $Y > j$ by $e^{\beta_l}$) for every cut-point $j$.

### Ordinal probit model

The **ordinal probit** model takes $F = \Phi$, the standard normal CDF:

$$
\Phi^{-1}\bigl(\Prob(Y \le j\mid\bx)\bigr) = \theta_j - \bbeta\T\bx,
\qquad j = 1,\ldots,J-1 .
$$

It arises from a latent-variable model $Y^{*} = \bbeta\T\bx + \varepsilon$
with $\varepsilon \sim \mathcal{N}(0,1)$, where the observed category is
$Y = j$ when $\theta_{j-1} < Y^{*} \le \theta_j$. Replacing the normal error
with a standard logistic error gives the ordinal logit model.

## Log-likelihood

From {eq}`eq-cumulative-link`, the probability of each category is a
difference of consecutive cumulative probabilities:

$$
\Prob(Y = j\mid\bx) = F(\theta_j - \bbeta\T\bx) - F(\theta_{j-1} - \bbeta\T\bx),
\qquad j = 1,\ldots,J,
$$

with the conventions $\theta_0 = -\infty$ and $\theta_J = +\infty$. For
observations $(\bx_i, y_i)$, $i = 1,\ldots,n$, the log-likelihood is

$$
\ell(\bbeta,\boldsymbol\theta)
= \sum_{i=1}^n\sum_{j=1}^J \mathbb{I}(y_i = j)\,
  \ln\bigl[F(\theta_j - \bbeta\T\bx_i) - F(\theta_{j-1} - \bbeta\T\bx_i)\bigr],
$$

where $\mathbb{I}(\cdot)$ is the indicator function and
$\boldsymbol\theta = (\theta_1,\ldots,\theta_{J-1})\T$. For the probit model,
$F = \Phi$ is the standard normal CDF, $\Phi(x) = \Prob(Z \le x)$ for
$Z\sim\mathcal{N}(0,1)$; for the logit model,
$F(x) = 1/(1+e^{-x})$. The parameters are estimated by maximizing
$\ell$ numerically.

## Summary and Common Mistakes
### Summary

- Cumulative link models describe $\Prob(Y\le j\mid\bx)$ with ordered
  cut-points $\theta_j$ and a common $\bbeta$.
- With the logit link, $e^{-\beta_l}$ is a cumulative odds ratio that is the
  same for every cut-point (proportional odds).
- With the probit link, the model corresponds to a latent normal variable
  cut at the thresholds.
- Category probabilities are differences of consecutive cumulative
  probabilities, which yields the log-likelihood.

### Common mistakes

- Including an intercept in $\bx$ in addition to the cut-points; the model
  is then not identifiable.
- Ignoring the sign convention. Software differs: some packages use
  $\theta_j - \bbeta\T\bx$ (as here), others $\theta_j + \bbeta\T\bx$.
- Assuming proportional odds without checking it; when the effect of a
  predictor differs across cut-points, a more flexible model is needed.

## References

- Agresti, A. (2013). *Categorical data analysis* (3rd ed.). Wiley.
- Frank, E., & Hall, M. (2001). A simple approach to ordinal classification. In L. De Raedt & P. Flach (Eds.), *Machine learning: ECML 2001* (pp. 145–156). Springer. <https://doi.org/10.1007/3-540-44795-4_13>
- UCLA Statistical Methods and Data Analytics. (n.d.). *Ordinal logistic regression: R data analysis examples*. <https://stats.oarc.ucla.edu/r/dae/ordinal-logistic-regression/>
- Wikipedia contributors. (n.d.). Ordinal regression. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Ordinal_regression>

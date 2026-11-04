# Prediction and Model Assessment

## Overview

A fitted regression model is used to estimate mean responses and to predict
new observations, and it must be checked before its conclusions are
trusted. This chapter derives confidence and prediction intervals,
introduces goodness-of-fit measures, and presents residual and influence
diagnostics. It builds on the hat matrix and sampling distributions ({doc}`09-ols-properties-and-geometry`) and on sums of squares and $t$ statistics ({doc}`10-anova-and-inference`).

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- construct a confidence interval for a mean response and a prediction
  interval for a new observation, and explain why the latter is wider;
- compute and interpret $R^2$, adjusted $R^2$, PRESS, and predicted $R^2$;
- compute standardized and studentized residuals and leverages; and
- use Cook's distance to identify influential observations.
:::

## Prediction at a new point

Let $\bx_0 = (1, x_{01}, \ldots, x_{0k})\T \in \R^p$ be the predictor values
of a new observation, with response

$$
\begin{gathered}
y_0 = \bx_0\T\bbeta + \varepsilon_0, \\
\varepsilon_0 \sim \mathcal{N}(0,\sigma^2),
\end{gathered}
$$

where $\varepsilon_0$ is independent of the errors in the data used to fit
the model. The point estimate of both the mean response
$\mu_0 = \E[y_0] = \bx_0\T\bbeta$ and the new response $y_0$ is the fitted
value

$$
\widehat{y}_0 = \bx_0\T\widehat{\bbeta}
\ \sim\ \mathcal{N}\bigl(\bx_0\T\bbeta,\ \sigma^2\,\bx_0\T(\bX\T\bX)^{-1}\bx_0\bigr),
$$

which follows from {prf:ref}`thm-ols-distribution` and
{prf:ref}`lem-linear-transform`. In particular, $\widehat{y}_0$ is an
unbiased estimator of $\mu_0$.

### Confidence interval for the mean response

Replacing $\sigma$ by $\widehat{\sigma}$ and using the independence of
$\widehat{\bbeta}$ and $\widehat{\sigma}^2$,

$$
\frac{\widehat{y}_0 - \bx_0\T\bbeta}{\widehat{\sigma}\sqrt{\bx_0\T(\bX\T\bX)^{-1}\bx_0}} \ \sim\ t(n-p).
$$

Hence a $100(1-\alpha)\%$ **confidence interval (CI)** for $\mu_0$ is

$$
\widehat{y}_0 \pm t_{1-\alpha/2,\,n-p}\ \widehat{\sigma}\sqrt{\bx_0\T(\bX\T\bX)^{-1}\bx_0}.
$$ (eq-ci-mean)

### Prediction interval for a new observation

A **prediction interval (PI)** $[l, u]$ for $y_0$ satisfies
$\Prob(l \le y_0 \le u) = 1-\alpha$. The prediction error is

$$
y_0 - \widehat{y}_0
= \varepsilon_0 - \bx_0\T(\widehat{\bbeta}-\bbeta)
\ \sim\ \mathcal{N}\bigl(0,\ \sigma^2\bigl[1 + \bx_0\T(\bX\T\bX)^{-1}\bx_0\bigr]\bigr),
$$

because the new error $\varepsilon_0$ is independent of $\widehat{\bbeta}$,
so the two variances add. Standardizing with $\widehat{\sigma}$,

$$
\frac{y_0 - \widehat{y}_0}{\widehat{\sigma}\sqrt{1 + \bx_0\T(\bX\T\bX)^{-1}\bx_0}} \ \sim\ t(n-p),
$$

and the $100(1-\alpha)\%$ prediction interval is

$$
\widehat{y}_0 \pm t_{1-\alpha/2,\,n-p}\ \widehat{\sigma}\sqrt{1 + \bx_0\T(\bX\T\bX)^{-1}\bx_0}.
$$ (eq-pi)

:::{prf:remark}
:label: rem-pi-wider

The prediction interval {eq}`eq-pi` is always wider than the confidence
interval {eq}`eq-ci-mean` at the same $\bx_0$: the CI accounts only for
the uncertainty in $\widehat{\bbeta}$, whereas the PI also includes the
variability $\sigma^2$ of a single new observation around its mean. As
$n\to\infty$ the CI shrinks to a point, but the PI does not.
:::

## Goodness of fit

### Coefficient of determination

The **coefficient of (multiple) determination** is

$$
\begin{aligned}
R^2 &= \frac{\mathit{SS}_R}{\mathit{SS}_T} \\
&= 1 - \frac{\mathit{SS}_E}{\mathit{SS}_T}.
\end{aligned}
$$

For a model with an intercept:

- $R^2$ is the proportion of the observed variability of $y$ explained by
  the predictors $x_1,\ldots,x_k$;
- $0 \le R^2 \le 1$, with $R^2 = 1$ when all residuals are zero; and
- adding a predictor never decreases $R^2$, whether or not the predictor is
  useful, so $R^2$ alone cannot be used to compare models of different
  sizes.

### Adjusted $R^2$

The **adjusted $R^2$** replaces sums of squares with mean squares, which
penalizes additional coefficients:

$$
\begin{aligned}
R^2_{\mathrm{adj}} &= 1 - \frac{\mathit{SS}_E/(n-p)}{\mathit{SS}_T/(n-1)} \\
&= 1 - \frac{n-1}{n-p}\bigl(1-R^2\bigr) \\
&= \frac{(n-1)R^2 - k}{n-k-1}.
\end{aligned}
$$

Properties:

- $R^2_{\mathrm{adj}} \le R^2 \le 1$; unlike $R^2$, the adjusted value can
  be negative when the model fits poorly;
- $R^2_{\mathrm{adj}}$ increases when a predictor is added only if the
  reduction in $\mathit{SS}_E$ outweighs the loss of one residual degree of
  freedom; and
- a large gap between $R^2$ and $R^2_{\mathrm{adj}}$ suggests that the model
  contains predictors that contribute little.

### PRESS and predicted $R^2$

Leave-one-out cross-validation (LOOCV) refits the model $n$ times, each time
leaving out one observation. Let $\widehat{y}_{(i)}$ be the prediction of
$y_i$ from the model fitted without observation $i$. The **prediction error
sum of squares** is

$$
\begin{aligned}
\mathrm{PRESS} &= \sum_{i=1}^n \bigl(y_i - \widehat{y}_{(i)}\bigr)^2 \\
&= \sum_{i=1}^n \left(\frac{e_i}{1-h_{ii}}\right)^2,
\end{aligned}
$$

where the second form, which avoids refitting, uses the leverages $h_{ii}$
defined below. The **predicted $R^2$** is

$$
\begin{gathered}
R^2_{\mathrm{pred}} = 1 - \frac{\mathrm{PRESS}}{\mathit{SS}_T}, \\
\mathit{SS}_T = \sum_{i=1}^n (y_i-\bar{y})^2 .
\end{gathered}
$$

It measures how well the model predicts observations it was not fitted to.

## Residual analysis

### Leverage

The diagonal entries of the hat matrix {eq}`eq-hat-matrix`,

$$
h_{ii} = \bx_i\T(\bX\T\bX)^{-1}\bx_i, \qquad i = 1,\ldots,n,
$$

are called **leverages**, where $\bx_i\T = (1, x_{i1},\ldots,x_{ik})$ is row
$i$ of $\bX$. They satisfy $0 \le h_{ii} \le 1$ and
$\sum_i h_{ii} = \tr(\bH) = p$. A large $h_{ii}$ means that observation $i$
has predictor values far from the center of the data.

### Types of residuals

Since $\be = (\bI_n-\bH)\by$ and $\bI_n-\bH$ is symmetric and idempotent,
$\Cov(\be) = \sigma^2(\bI_n-\bH)$ and therefore
$\Var(e_i) = \sigma^2(1-h_{ii})$. This motivates three scalings.

Ordinary residuals
: $e_i = y_i - \widehat{y}_i$.

Standardized residuals
: $d_i = \dfrac{e_i}{\widehat{\sigma}}$, which are approximately
  $\mathcal{N}(0,1)$ when the model is correct and $n$ is large.

Studentized residuals
: $r_i = \dfrac{e_i}{\sqrt{\widehat{\sigma}^2(1-h_{ii})}}$, which divide by
  the estimated standard deviation of each residual. These are also called
  internally studentized residuals.

Because $0 < 1-h_{ii} \le 1$, we have $|r_i| \ge |d_i|$. Studentized
residuals correct for the smaller variance of residuals at high-leverage
points and are therefore better for detecting outliers; values with
$|r_i| > 3$ deserve attention.

### Residual plots

Plot residuals against fitted values and against each predictor. A
patternless horizontal band supports the model; curvature suggests a
missing nonlinear term, and a funnel shape suggests non-constant variance.
A normal quantile–quantile (Q–Q) plot of the residuals checks the
normality assumption ({doc}`04-goodness-of-fit-tests`).

## Influential observations

An **influential observation** is one whose removal changes the fitted model
substantially: parameter estimates or predictions depend more on a small
subset of the data than on the majority. Locating such observations matters
for two reasons:

- if they are erroneous values, they should be corrected or removed; and
- if they are valid, they control key properties of the model, and the
  analyst should know this.

### Cook's distance

Cook's distance (Cook, 1977) compares the fitted values with and without
observation $i$. Let $\widehat{\bbeta}_{(i)}$ be the OLS estimator computed
without observation $i$, and $\widehat{\by}_{(i)} = \bX\widehat{\bbeta}_{(i)}$.
The squared Euclidean distance between the two fitted vectors is

$$
\lVert\widehat{\by}_{(i)}-\widehat{\by}\rVert_2^2
= \bigl(\widehat{\bbeta}_{(i)}-\widehat{\bbeta}\bigr)\T\bX\T\bX\bigl(\widehat{\bbeta}_{(i)}-\widehat{\bbeta}\bigr).
$$

Scaling by $p\,\widehat{\sigma}^2$ gives **Cook's distance**

$$
\begin{aligned}
D_i &= \frac{\bigl(\widehat{\bbeta}_{(i)}-\widehat{\bbeta}\bigr)\T\bX\T\bX\bigl(\widehat{\bbeta}_{(i)}-\widehat{\bbeta}\bigr)}{p\,\widehat{\sigma}^2} \\
&= \frac{r_i^2}{p}\cdot\frac{h_{ii}}{1-h_{ii}}, \qquad i = 1,\ldots,n,
\end{aligned}
$$ (eq-cooks-distance)

where $r_i$ is the studentized residual. The second form shows that $D_i$
combines two components: $r_i^2$, which reflects how poorly the model fits
observation $i$, and $h_{ii}/(1-h_{ii})$, which measures how far the point
lies from the rest of the data in predictor space.

A common rule of thumb flags observation $i$ as influential when $D_i > 1$.

## Summary and Common Mistakes
### Summary

- The CI for a mean response has half-width
  $t_{1-\alpha/2,n-p}\,\widehat{\sigma}\sqrt{\bx_0\T(\bX\T\bX)^{-1}\bx_0}$;
  the PI adds $1$ under the square root.
- $R^2$ never decreases with more predictors; $R^2_{\mathrm{adj}}$ and
  $R^2_{\mathrm{pred}}$ penalize unhelpful predictors.
- $\Var(e_i) = \sigma^2(1-h_{ii})$; studentized residuals account for
  leverage.
- Cook's distance combines residual size and leverage; $D_i > 1$ is a common
  warning threshold.

### Common mistakes

- Using a confidence interval when the question concerns a single future
  observation; that requires the wider prediction interval.
- Choosing between models by $R^2$; it always favors the larger model.
- Treating a large ordinary residual as an outlier without accounting for
  its leverage.
- Deleting influential points automatically. Influence is a reason to
  investigate, not a reason to delete.

## References

- Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley.
- Cook, R. D. (1977). Detection of influential observation in linear regression. *Technometrics, 19*(1), 15–18. <https://doi.org/10.1080/00401706.1977.10489493>
- He, Z. [何志坚]. (n.d.). *数理统计讲义* [Lecture notes on mathematical statistics]. <https://bookdown.org/hezhijian/book/>

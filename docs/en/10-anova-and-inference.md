# ANOVA and Statistical Inference

## Overview

The total variation of the response can be split into a part explained by
the regression and a residual part. This **analysis of variance (ANOVA)**
underlies the $F$-tests for the overall model and for groups of
coefficients. Together with the $t$-test for a single coefficient, these
tests answer the question of which predictors contribute to the model. It builds on the orthogonality and sampling results of {doc}`09-ols-properties-and-geometry` ({prf:ref}`prop-orthogonality`, {prf:ref}`thm-ols-distribution`) and on the $t$, $\chi^2$, and $F$ distributions.

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- prove the decomposition $\mathit{SS}_T = \mathit{SS}_R + \mathit{SS}_E$
  for models with and without an intercept;
- construct the ANOVA table and state the degrees of freedom of each sum
  of squares;
- carry out $t$-tests and confidence intervals for individual coefficients;
  and
- test the overall regression and a subset of coefficients with $F$-tests.
:::

## Notation

| Symbol | Meaning |
|---|---|
| $\bar{y}$ | sample mean of the responses, $\frac1n\sum_i y_i$ |
| $\mathit{SS}_T$ | total sum of squares |
| $\mathit{SS}_R$ | regression sum of squares |
| $\mathit{SS}_E$ | residual (error) sum of squares |
| $\mathit{MS}_R$, $\mathit{MS}_E$ | mean squares: sum of squares divided by its degrees of freedom |
| $c_{jj}$ | $j$-th diagonal entry of $(\bX\T\bX)^{-1}$, $j=0,\ldots,k$ |
| $t_{q,\nu}$ | $q$-quantile of the $t$ distribution with $\nu$ degrees of freedom |
| $F_{q;\nu_1,\nu_2}$ | $q$-quantile of the $F$ distribution with $(\nu_1,\nu_2)$ degrees of freedom |

## Variance decomposition

### Models with an intercept

:::{prf:theorem} Sum of squares decomposition
:label: thm-ss-decomposition

For a linear model that contains an intercept,

$$
\underbrace{\sum_{i=1}^n (y_i-\bar{y})^2}_{\mathit{SS}_T}
= \underbrace{\sum_{i=1}^n (\widehat{y}_i-\bar{y})^2}_{\mathit{SS}_R}
+ \underbrace{\sum_{i=1}^n (y_i-\widehat{y}_i)^2}_{\mathit{SS}_E} .
$$ (eq-ss-decomposition)

That is, total sum of squares = regression sum of squares + residual sum of
squares.
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

Write $y_i - \bar{y} = (y_i - \widehat{y}_i) + (\widehat{y}_i - \bar{y})$
and expand the square:

$$
\sum_{i=1}^n (y_i-\bar{y})^2
= \sum_{i=1}^n (y_i-\widehat{y}_i)^2 + \sum_{i=1}^n (\widehat{y}_i-\bar{y})^2
+ 2\sum_{i=1}^n (\widehat{y}_i-\bar{y})(y_i-\widehat{y}_i).
$$

The cross term vanishes:

$$
\begin{aligned}
\sum_{i=1}^n (\widehat{y}_i-\bar{y})\,e_i &= \underbrace{\widehat{\by}\T\be}_{=0} - \bar{y}\underbrace{\sum_{i=1}^n e_i}_{=0} \\
&= 0,
\end{aligned}
$$

by {prf:ref}`prop-orthogonality`; the second equality uses the intercept.
:::

Useful computational forms are

$$
\begin{aligned}
\mathit{SS}_T &= \sum_{i=1}^n y_i^2 - n\bar{y}^2 \\
             &= \by\T\by - \tfrac{1}{n}\Bigl(\sum_{i=1}^n y_i\Bigr)^2, \\
\mathit{SS}_R &= \widehat{\bbeta}\T\bX\T\by - \tfrac{1}{n}\Bigl(\sum_{i=1}^n y_i\Bigr)^2, \\
\mathit{SS}_E &= (\by-\bX\widehat{\bbeta})\T(\by-\bX\widehat{\bbeta}) \\
             &= \by\T\by - \widehat{\bbeta}\T\bX\T\by .
\end{aligned}
$$

The last identity uses the normal equations,
$\widehat{\bbeta}\T\bX\T\bX\widehat{\bbeta} = \widehat{\bbeta}\T\bX\T\by$.
The degrees of freedom of $\mathit{SS}_T$, $\mathit{SS}_R$, and
$\mathit{SS}_E$ are $n-1$, $p-1 = k$, and $n-p$, respectively; they add up
in the same way as the sums of squares.

### Models without an intercept

For the no-intercept model

$$
y_i = \beta_1x_{i1} + \beta_2x_{i2} + \cdots + \beta_kx_{ik} + \varepsilon_i,
\qquad i = 1,\ldots,n,
$$

the fitted hyperplane passes through the origin rather than through the
point of means $(\bar{\bx},\bar{y})$, and the residuals need not sum to
zero. The decomposition therefore uses uncentered sums of squares:

$$
\underbrace{\sum_{i=1}^n y_i^2}_{\mathit{SS}_T}
= \underbrace{\sum_{i=1}^n \widehat{y}_i^{\,2}}_{\mathit{SS}_R}
+ \underbrace{\sum_{i=1}^n (y_i-\widehat{y}_i)^2}_{\mathit{SS}_E} .
$$ (eq-ss-decomposition-noint)

The cross term $2\sum_i \widehat{y}_i e_i = 2\,\widehat{\by}\T\be$ is zero
by {prf:ref}`prop-orthogonality`, which does not require an intercept. In
matrix form,

$$
\begin{aligned}
\mathit{SS}_T &= \by\T\by, \\
\mathit{SS}_R &= \widehat{\bbeta}\T\bX\T\bX\widehat{\bbeta} \\
&= \widehat{\bbeta}\T\bX\T\by, \\
\mathit{SS}_E &= \by\T\by - \widehat{\bbeta}\T\bX\T\by ,
\end{aligned}
$$

with $n$, $k$, and $n-k$ degrees of freedom (here $p = k$).

:::{warning}
Because the no-intercept $\mathit{SS}_T$ is not centered, the $R^2$
reported for a no-intercept model is not comparable with the $R^2$ of a
model with an intercept.
:::

## ANOVA tables

### Simple linear regression

For $y_i = \beta_0 + \beta_1x_i + \varepsilon_i$, with $S_{xx}$ and $S_{xy}$
as defined in {doc}`08-ols-and-ml-estimation`:

| Source of variation | Sum of squares | Degrees of freedom | Mean square |
|---|---|---|---|
| Regression | $\mathit{SS}_R = \widehat{\beta}_1S_{xy}$ | $1$ | $\mathit{MS}_R = \mathit{SS}_R/1$ |
| Error | $\mathit{SS}_E = \mathit{SS}_T - \widehat{\beta}_1S_{xy}$ | $n-2$ | $\mathit{MS}_E = \mathit{SS}_E/(n-2)$ |
| Total | $\mathit{SS}_T$ | $n-1$ | |

The expected mean squares are

$$
\begin{gathered}
\E[\mathit{MS}_E] = \sigma^2, \\
\E[\mathit{MS}_R] = \sigma^2 + \beta_1^2S_{xx},
\end{gathered}
$$

so $\mathit{MS}_R/\mathit{MS}_E$ tends to exceed one when $\beta_1 \ne 0$.

### Multiple linear regression with an intercept

| Source of variation | Sum of squares | Degrees of freedom | Mean square |
|---|---|---|---|
| Regression | $\sum_{i}(\widehat{y}_i-\bar{y})^2$ | $k$ | $\mathit{MS}_R = \mathit{SS}_R/k$ |
| Error | $\sum_{i}(y_i-\widehat{y}_i)^2$ | $n-k-1$ | $\mathit{MS}_E = \mathit{SS}_E/(n-k-1)$ |
| Total | $\sum_{i}(y_i-\bar{y})^2$ | $n-1$ | |

### Multiple linear regression without an intercept

| Source of variation | Sum of squares | Degrees of freedom | Mean square |
|---|---|---|---|
| Regression | $\sum_{i}\widehat{y}_i^{\,2}$ | $k$ | $\mathit{MS}_R = \mathit{SS}_R/k$ |
| Error | $\sum_{i}(y_i-\widehat{y}_i)^2$ | $n-k$ | $\mathit{MS}_E = \mathit{SS}_E/(n-k)$ |
| Total | $\sum_{i}y_i^2$ | $n$ | |

### Distributions of the sums of squares

:::{prf:theorem} Distributions in the ANOVA table
:label: thm-anova-distributions

Under {prf:ref}`asm-classical` for a model with an intercept:

1. $\mathit{SS}_E/\sigma^2 \sim \chi^2(n-p)$ (from {prf:ref}`thm-ols-distribution`);
2. $\mathit{SS}_R$, $\mathit{SS}_E$, and $\bar{y}$ are mutually independent;
   and
3. if $H_0:\beta_1 = \cdots = \beta_k = 0$ holds, then
   $\mathit{SS}_R/\sigma^2 \sim \chi^2(k)$.
:::

Part 3 holds only under the null hypothesis; when some slope is nonzero,
$\mathit{SS}_R/\sigma^2$ follows a noncentral $\chi^2$ distribution. The
same statements hold for the no-intercept model with $\mathit{SS}_E/\sigma^2
\sim \chi^2(n-k)$ and, under $H_0$, $\mathit{SS}_R/\sigma^2 \sim \chi^2(k)$.

## Inference for a single coefficient

### The $t$-test

To test whether coefficient $\beta_j$ equals a specified value $\beta_{j,0}$
(usually $0$),

$$
\begin{gathered}
H_0 :\ \beta_j = \beta_{j,0} \quad\text{versus} \\
H_1 :\ \beta_j \ne \beta_{j,0}, \qquad j = 0,1,\ldots,k,
\end{gathered}
$$

use the statistic

$$
\begin{aligned}
T_j &= \frac{\widehat{\beta}_j - \beta_{j,0}}{\mathrm{SE}(\widehat{\beta}_j)} \\
&= \frac{\widehat{\beta}_j - \beta_{j,0}}{\widehat{\sigma}\sqrt{c_{jj}}} \ \sim\ t(n-p) \quad\text{under } H_0,
\end{aligned}
$$ (eq-t-test)

where $\mathrm{SE}(\widehat{\beta}_j) = \sqrt{\widehat{\sigma}^2c_{jj}}$ is
the **standard error** of $\widehat{\beta}_j$, i.e. the estimated standard
deviation of the estimator. The $t$ distribution follows from
{prf:ref}`thm-ols-distribution`: the numerator divided by
$\sigma\sqrt{c_{jj}}$ is standard normal, it is independent of
$\widehat{\sigma}^2$, and $(n-p)\widehat{\sigma}^2/\sigma^2 \sim \chi^2(n-p)$.

At significance level $\alpha$, reject $H_0$ when the observed value $t_j$
satisfies $|t_j| > t_{1-\alpha/2,\,n-p}$.

### Confidence interval

A $100(1-\alpha)\%$ confidence interval for $\beta_j$ is

$$
\widehat{\beta}_j - t_{1-\alpha/2,\,n-p}\,\mathrm{SE}(\widehat{\beta}_j)
\ \le\ \beta_j\ \le\
\widehat{\beta}_j + t_{1-\alpha/2,\,n-p}\,\mathrm{SE}(\widehat{\beta}_j).
$$

## The overall $F$-test

The **overall $F$-test** asks whether the regression explains any
variation at all, i.e. whether at least one slope differs from zero:

$$
\begin{gathered}
H_0 :\ \beta_1 = \beta_2 = \cdots = \beta_k = 0 \quad\text{versus} \\
H_1 :\ \beta_j \ne 0 \text{ for at least one } j \in \{1,\ldots,k\}.
\end{gathered}
$$

The null hypothesis concerns the slopes only; the intercept $\beta_0$ is not
tested. By {prf:ref}`thm-anova-distributions`,

$$
\begin{aligned}
F &= \frac{\mathit{MS}_R}{\mathit{MS}_E} \\
&= \frac{\mathit{SS}_R/(p-1)}{\mathit{SS}_E/(n-p)} \ \sim\ F(p-1,\ n-p) \quad\text{under } H_0 .
\end{aligned}
$$ (eq-overall-f)

Reject $H_0$ at level $\alpha$ when the observed value $f$ exceeds
$F_{1-\alpha;\,p-1,\,n-p}$.

## Testing a subset of coefficients

### Extra sum of squares

Partition the coefficients as $\bbeta = (\bbeta_1\T, \bbeta_2\T)\T$, where
$\bbeta_1$ contains the $r$ coefficients to be tested and $\bbeta_2$ the
remaining $p-r$ coefficients (including the intercept), and partition
$\bX = [\bX_1,\ \bX_2]$ accordingly.

For a fitted model, define the **uncorrected regression sum of squares**

$$
\begin{aligned}
\mathit{SS}_R(\bbeta) &= \widehat{\bbeta}\T\bX\T\by \\
&= \mathit{SS}_R + \frac{1}{n}\Bigl(\sum_{i=1}^n y_i\Bigr)^2 .
\end{aligned}
$$

Let $\widehat{\bbeta}_2^{(0)}$ denote the OLS estimator in the reduced model
$\by = \bX_2\bbeta_2 + \bvarepsilon$. The **extra sum of squares** due to
$\bbeta_1$ given $\bbeta_2$ is the increase in the regression sum of squares
when $\bX_1$ is added to the reduced model:

$$
\begin{aligned}
\mathit{SS}_R(\bbeta_1 \mid \bbeta_2)
&= \mathit{SS}_R(\bbeta) - \mathit{SS}_R(\bbeta_2) \\
 &= \widehat{\bbeta}\T\bX\T\by - \bigl(\widehat{\bbeta}_2^{(0)}\bigr)\T\bX_2\T\by \\
&= \underbrace{\Bigl[\mathit{SS}_R(\bbeta) - \tfrac{1}{n}\bigl(\textstyle\sum_i y_i\bigr)^2\Bigr]}_{\mathit{SS}_R \text{ of the full model}}
 - \underbrace{\Bigl[\mathit{SS}_R(\bbeta_2) - \tfrac{1}{n}\bigl(\textstyle\sum_i y_i\bigr)^2\Bigr]}_{\mathit{SS}_R \text{ of the reduced model}} .
\end{aligned}
$$

Equivalently, $\mathit{SS}_R(\bbeta_1\mid\bbeta_2)$ equals the reduction in
$\mathit{SS}_E$ achieved by adding $\bX_1$.

### Partial $F$-test

To test

$$
\begin{gathered}
H_0 :\ \bbeta_1 = \bzero \quad\text{versus} \\
H_1 :\ \bbeta_1 \ne \bzero ,
\end{gathered}
$$

use

$$
\begin{aligned}
F_r &= \frac{\mathit{SS}_R(\bbeta_1\mid\bbeta_2)/r}{\mathit{SS}_E/(n-p)} \\
&= \frac{\mathit{SS}_R(\bbeta_1\mid\bbeta_2)/r}{\mathit{MS}_E} \ \sim\ F(r,\ n-p) \quad\text{under } H_0,
\end{aligned}
$$ (eq-partial-f)

where the numerator has $r$ degrees of freedom (one per tested coefficient)
and $\mathit{SS}_E$ comes from the full model with $n-p$ degrees of
freedom. If the observed value exceeds $F_{1-\alpha;\,r,\,n-p}$, reject $H_0$
and conclude that at least one coefficient in $\bbeta_1$ is nonzero, i.e. at
least one of the predictors in $\bX_1$ contributes significantly given the
others. With $r = 1$, the partial $F$-test is equivalent to the two-sided
$t$-test {eq}`eq-t-test`, since $F_1 = T_j^2$.

:::{prf:example} Sequential sums of squares
:label: ex-sequential-ss

For a model with four predictors, the regression sum of squares can be
built up by adding one predictor at a time:

$$
\begin{aligned}
\mathit{SS}_R &= \mathit{SS}_R(\beta_1,\beta_2,\beta_3,\beta_4 \mid \beta_0) \\
&= \mathit{SS}_R(\beta_1\mid\beta_0)
 + \mathit{SS}_R(\beta_2\mid\beta_1,\beta_0)
 + \mathit{SS}_R(\beta_3\mid\beta_2,\beta_1,\beta_0)
 + \mathit{SS}_R(\beta_4\mid\beta_3,\beta_2,\beta_1,\beta_0).
\end{aligned}
$$

Each term measures the contribution of one predictor given those entered
before it, so the terms depend on the order of entry.
:::

## Summary and Common Mistakes
### Summary

- With an intercept,
  $\mathit{SS}_T = \mathit{SS}_R + \mathit{SS}_E$ with $n-1 = k + (n-p)$
  degrees of freedom; without an intercept, uncentered sums of squares
  decompose with $n = k + (n-k)$.
- $T_j = (\widehat{\beta}_j-\beta_{j,0})/\mathrm{SE}(\widehat{\beta}_j) \sim t(n-p)$
  tests one coefficient.
- $F = \mathit{MS}_R/\mathit{MS}_E \sim F(p-1,n-p)$ tests all slopes
  jointly.
- The partial $F$-test compares nested models through the extra sum of
  squares.

### Common mistakes

- Stating that $\mathit{SS}_R/\sigma^2 \sim \chi^2(k)$ unconditionally; it
  holds only under $H_0$.
- Reading a significant overall $F$-test as "all coefficients are nonzero".
  It indicates that at least one slope is nonzero.
- Interpreting sequential sums of squares without noting that they depend
  on the order in which predictors enter.
- Comparing $R^2$ of an intercept model with that of a no-intercept model.

## References

- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An introduction to statistical learning: With applications in R*. Springer. <https://doi.org/10.1007/978-1-4614-7138-7>
- Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley.
- He, Z. [何志坚]. (n.d.). *数理统计讲义* [Lecture notes on mathematical statistics]. <https://bookdown.org/hezhijian/book/>
- Zaiontz, C. (n.d.). *Multiple regression without intercept*. Real Statistics Using Excel. <https://real-statistics.com/multiple-regression/multiple-regression-without-intercept>

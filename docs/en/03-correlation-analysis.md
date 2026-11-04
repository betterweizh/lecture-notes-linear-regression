# Correlation Analysis

## Overview

Before fitting a regression, it is often useful to measure how strongly two
variables are associated. This chapter presents the Pearson correlation
coefficient for linear association between continuous variables, the rank
correlation coefficients of Spearman and Kendall for monotone association
and ordinal data, and Goodman and Kruskal's $\gamma$ for contingency tables.
For each it gives the definition, the main properties, and a test of no
association. Code is in {doc}`23-correlation-and-goodness-of-fit-code`. It builds on covariance and correlation ({doc}`01-probability-and-random-vectors`) and on the $t$ and normal distributions used in hypothesis tests ({doc}`02-sampling-distributions-and-limit-theorems`).

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- compute and interpret the Pearson correlation coefficient, and relate it
  to simple linear regression;
- test $H_0:\rho = 0$ with the $t$-test and $H_0:\rho = \rho_0$ with Fisher's
  $z$-transformation, and build a confidence interval for $\rho$;
- compute Spearman's and Kendall's rank correlation coefficients, with and
  without ties, and test them for significance; and
- measure association in a contingency table with Goodman and Kruskal's
  $\gamma$.
:::

## Notation

| Symbol | Meaning |
|---|---|
| $(x_i, y_i)$, $i = 1,\ldots,n$ | paired observations of $X$ and $Y$ |
| $S_{xx}$, $S_{yy}$, $S_{xy}$ | corrected sums of squares and cross-products |
| $\rho$, $r$ | population and sample Pearson correlation |
| $R_S$ | Spearman's rank correlation coefficient |
| $\tau$ | Kendall's rank correlation coefficient |
| $n_c$, $n_d$ | numbers of concordant and discordant pairs |

## Pearson correlation coefficient

### Definition

The population correlation of $X$ and $Y$ is
$\rho = \Cov(X,Y)/\sqrt{\Var(X)\Var(Y)}$
({doc}`01-probability-and-random-vectors`). Its sample counterpart, the
**Pearson (or Bravais–Pearson) correlation coefficient**, is

$$
\begin{aligned}
r &= \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}
         {\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2\,\sum_{i=1}^n (y_i - \bar{y})^2}} \\
  &= \frac{S_{xy}}{\sqrt{S_{xx}S_{yy}}},
\end{aligned}
$$ (eq-pearson-r)

where

$$
\begin{aligned}
S_{xx} &= \sum_{i=1}^n (x_i - \bar{x})^2, \\
S_{yy} &= \sum_{i=1}^n (y_i - \bar{y})^2, \\
S_{xy} &= \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}) \\
&= \sum_{i=1}^n x_iy_i - n\bar{x}\bar{y} .
\end{aligned}
$$

It is appropriate when both variables are measured on a continuous scale
and the relationship is approximately linear.

### Properties

- $-1 \le r \le 1$. The value $r = 1$ ($r = -1$) occurs exactly when the
  points lie on a straight line with positive (negative) slope; the sign
  gives the direction of the association.
- $r$ is symmetric, $r(X,Y) = r(Y,X)$.
- $r$ does not depend on the units of measurement: it is unchanged by
  $x \mapsto a + bx$ and $y \mapsto c + dy$ with $bd > 0$ (the sign flips
  if $bd < 0$). Heights in metres or centimetres give the same $r$.
- $r$ measures **linear** association only. A value near zero means that
  there is no linear relationship; the variables may still be strongly
  related in a nonlinear way, and uncorrelated variables need not be
  independent.

### Relation to simple linear regression

In the simple linear regression of $y$ on $x$
({doc}`08-ols-and-ml-estimation`), the slope estimate and the coefficient of
determination ({doc}`13-prediction-and-model-assessment`) are

$$
\begin{aligned}
\widehat{\beta}_1 &= \frac{S_{xy}}{S_{xx}} \\
&= r\,\frac{s_y}{s_x}, \\
R^2 &= r^2 ,
\end{aligned}
$$

where $s_x$ and $s_y$ are the sample standard deviations. Testing
$H_0:\rho = 0$ is therefore equivalent to testing $H_0:\beta_1 = 0$.

### $t$-test of zero correlation

Suppose $(X, Y)$ has a bivariate normal distribution. To test
$H_0:\rho = 0$, use

$$
T = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}} \sim t(n-2) \quad\text{under } H_0 .
$$ (eq-corr-t)

This is the $t$ statistic for $\widehat{\beta}_1$ in the simple linear
regression. Reject $H_0$ at level $\alpha$ when
$\lvert T\rvert > t_{1-\alpha/2,\,n-2}$.

### Fisher's $z$-transformation

To test $H_0:\rho = \rho_0$ for $\rho_0 \ne 0$, or to build a confidence
interval for $\rho$, use **Fisher's $z$-transformation**

$$
\begin{aligned}
z &= \operatorname{artanh}(r) \\
&= \frac12\ln\frac{1+r}{1-r} .
\end{aligned}
$$

For bivariate normal data, $z$ is approximately normal with mean
$\operatorname{artanh}(\rho)$ and standard deviation $1/\sqrt{n-3}$. Hence,
under $H_0:\rho = \rho_0$,

$$
Z = \bigl(\operatorname{artanh}(r) - \operatorname{artanh}(\rho_0)\bigr)\sqrt{n-3}
\ \overset{\text{approx.}}{\sim}\ \mathcal{N}(0,1),
$$ (eq-fisher-z)

and an approximate $100(1-\alpha)\%$ confidence interval for $\rho$ is

$$
\tanh\!\left(\operatorname{artanh}(r) \pm \frac{z_{1-\alpha/2}}{\sqrt{n-3}}\right).
$$

## Spearman's rank correlation coefficient

### Definition

Replace each $x_i$ by its rank among $x_1,\ldots,x_n$ (the smallest value
has rank 1) and each $y_i$ by its rank among the $y$'s. Tied values receive
the average of the ranks they occupy. **Spearman's rank correlation
coefficient** $R_S$ is the Pearson correlation {eq}`eq-pearson-r` of the
ranks,

$$
R_S = r\bigl(\operatorname{rank}(x), \operatorname{rank}(y)\bigr).
$$

When there are no ties, with $d_i = \operatorname{rank}(x_i) - \operatorname{rank}(y_i)$,
it simplifies to

$$
R_S = 1 - \frac{6\sum_{i=1}^n d_i^2}{n(n^2-1)} .
$$ (eq-spearman)

With ties, {eq}`eq-spearman` is only an approximation, and the definition
through the ranks should be used.

### Properties

- $-1 \le R_S \le 1$. $R_S = 1$ when the two rankings agree exactly and
  $R_S = -1$ when one is the reverse of the other.
- $R_S$ is unchanged by any strictly increasing transformation of $X$ or
  $Y$, so it measures **monotone**, not only linear, association, and it
  can be used for ordinal variables.
- $R_S$ is the same whether ranks are assigned in ascending or descending
  order, as long as both variables use the same order.

### Test of no association

Under $H_0$ of independence, $\E[R_S] = 0$ and $\Var(R_S) = 1/(n-1)$. For
large samples (roughly $n > 10$),

$$
Z = R_S\sqrt{n-1} \ \overset{\text{approx.}}{\sim}\ \mathcal{N}(0,1).
$$

For small samples, exact critical values or $p$-values of $R_S$ (or of
$\sum_i d_i^2$) are obtained from tables or by permutation.

## Kendall's rank correlation coefficient

### Concordant and discordant pairs

Consider two observations $(x_i, y_i)$ and $(x_j, y_j)$ with $i < j$. The
pair is

- **concordant** if $x_i - x_j$ and $y_i - y_j$ have the same sign;
- **discordant** if they have opposite signs; and
- **tied** if $x_i = x_j$ or $y_i = y_j$.

Let $n_c$ and $n_d$ be the numbers of concordant and discordant pairs, and
$n_0 = n(n-1)/2$ the total number of pairs.

### Definition

With no ties, **Kendall's $\tau$** is the difference between the
proportions of concordant and discordant pairs,

$$
\begin{aligned}
\tau &= \frac{n_c - n_d}{n_0} \\
     &= \frac{2}{n(n-1)}\sum_{i<j}\sign(x_i - x_j)\,\sign(y_i - y_j) .
\end{aligned}
$$ (eq-kendall)

The sum $S = n_c - n_d$ is the **Kendall score**. Like $R_S$, $\tau$ lies in
$[-1, 1]$, equals $1$ for identical rankings and $-1$ for reversed
rankings, and is close to zero when $X$ and $Y$ are independent.

### Ties: $\tau_A$, $\tau_B$, and $\tau_C$

A tied pair is neither concordant nor discordant, and several versions of
the coefficient handle ties differently.

- **$\tau_A = (n_c - n_d)/n_0$** makes no adjustment for ties; it equals
  {eq}`eq-kendall` when there are none.
- **$\tau_B$** corrects the denominator for ties:

  $$
  \tau_B = \frac{n_c - n_d}{\sqrt{(n_0 - n_1)(n_0 - n_2)}},
  $$

  where $n_1 = \sum_k t_k(t_k-1)/2$ and $n_2 = \sum_l u_l(u_l-1)/2$, and
  $t_k$ ($u_l$) is the number of tied values in the $k$-th ($l$-th) group of
  ties in $x$ ($y$). It is suitable for square contingency tables and is
  the version computed by most software.
- **$\tau_C$** (Stuart–Kendall) is intended for rectangular contingency
  tables with $r$ rows and $c$ columns:

  $$
  \tau_C = \frac{2(n_c - n_d)}{n^2(m-1)/m}, \qquad m = \min(r, c).
  $$

### Test of no association

Under $H_0$ of independence and no ties, $\E[\tau] = 0$ and
$\Var(\tau) = 2(2n+5)/\bigl(9n(n-1)\bigr)$. For large samples (roughly
$n > 10$),

$$
\begin{aligned}
Z &= \frac{\tau}{\sqrt{2(2n+5)/\bigl(9n(n-1)\bigr)}} \\
&= \frac{S}{\sqrt{n(n-1)(2n+5)/18}}
\ \overset{\text{approx.}}{\sim}\ \mathcal{N}(0,1).
\end{aligned}
$$

For small samples, exact critical values of $S$ are tabulated; with ties,
the variance of $S$ must be corrected.

## Goodman and Kruskal's $\gamma$

For two ordinal variables summarized in an $I\times J$ contingency table
with counts $n_{ij}$, the numbers of concordant and discordant pairs of
observations are

$$
\begin{gathered}
K = \sum_{i<k}\sum_{j<l} n_{ij}\,n_{kl}, \\
D = \sum_{i<k}\sum_{j>l} n_{ij}\,n_{kl},
\end{gathered}
$$

and **Goodman and Kruskal's $\gamma$** is

$$
\gamma = \frac{K - D}{K + D} .
$$

Tied pairs are ignored, so $-1 \le \gamma \le 1$, with the sign giving the
direction of the association and larger absolute values a stronger
association. Because ties are dropped, $\lvert\gamma\rvert \ge \lvert\tau_B\rvert$.

## Choosing a coefficient

| Coefficient | Measures | Data | Robust to outliers |
|---|---|---|---|
| Pearson $r$ | linear association | continuous, roughly bivariate normal for tests | no |
| Spearman $R_S$ | monotone association | continuous or ordinal | yes |
| Kendall $\tau$ | monotone association (pairs) | continuous or ordinal, small samples | yes |
| Goodman–Kruskal $\gamma$ | association in ordered tables | ordinal, grouped | yes |

## Summary and Common Mistakes
### Summary

- $r = S_{xy}/\sqrt{S_{xx}S_{yy}}$ measures linear association;
  $R^2 = r^2$ in simple linear regression.
- $T = r\sqrt{n-2}/\sqrt{1-r^2} \sim t(n-2)$ tests $\rho = 0$; Fisher's
  $z = \operatorname{artanh}(r)$ with standard error $1/\sqrt{n-3}$ tests
  $\rho = \rho_0$ and gives confidence intervals.
- Spearman's $R_S$ is the Pearson correlation of the ranks; Kendall's
  $\tau$ compares concordant and discordant pairs. Both measure monotone
  association and are robust to outliers.
- Goodman and Kruskal's $\gamma$ measures association in ordered
  contingency tables.

### Common mistakes

- Concluding that two variables are independent because $r \approx 0$; $r$
  only detects linear association.
- Reading a significant correlation as evidence of causation.
- Testing $H_0:\rho = \rho_0 \ne 0$ with the $t$-test {eq}`eq-corr-t`, which
  is valid only for $\rho_0 = 0$.
- Using the no-ties formula {eq}`eq-spearman` when there are many ties.

## References

- Heumann, C., Schomaker, M., & Shalabh. (2016). *Introduction to statistics and data analysis: With exercises, solutions and applications in R*. Springer. <https://doi.org/10.1007/978-3-319-46162-5>
- Hollander, M., Wolfe, D. A., & Chicken, E. (2014). *Nonparametric statistical methods* (3rd ed.). Wiley.
- Kanji, G. K. (2006). *100 statistical tests* (3rd ed.). Sage.
- Wikipedia contributors. (n.d.). Kendall rank correlation coefficient. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>

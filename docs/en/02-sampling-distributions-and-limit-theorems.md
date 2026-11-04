# Sampling Distributions and Limit Theorems

## Overview

Tests and confidence intervals in regression rest on a small family of
distributions derived from the normal: the $\chi^2$, $t$, and $F$
distributions and their noncentral versions. This chapter defines them,
states the results on quadratic forms (including Cochran's theorem) that
produce them in the linear model, reviews the basic ideas of estimation and
hypothesis testing, and summarizes the limit theorems behind the
large-sample results used for generalized linear models. It builds on {doc}`01-probability-and-random-vectors` and on rank, trace, and idempotent matrices ({doc}`05-linear-algebra`).

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- define the $\chi^2$, $t$, and $F$ distributions and their noncentral
  versions;
- determine the distribution of a quadratic form in normal variables and
  apply Cochran's theorem;
- decompose the mean squared error of an estimator into variance and
  squared bias;
- explain type I and type II errors, power, $p$-values, and confidence
  intervals; and
- state the law of large numbers, the central limit theorem, and the
  large-sample properties of maximum likelihood estimators.
:::

## Distributions derived from the normal

### The $\chi^2$ distribution

:::{prf:definition} Chi-square distribution
:label: def-chi2

If $Z_1,\ldots,Z_k$ are independent $\mathcal{N}(0,1)$ variables, then

$$
U = \sum_{i=1}^k Z_i^2 \sim \chi^2(k),
$$

the **$\chi^2$ distribution with $k$ degrees of freedom**. It has
$\E[U] = k$ and $\Var(U) = 2k$.
:::

If $U_1 \sim \chi^2(k_1)$ and $U_2 \sim \chi^2(k_2)$ are independent, then
$U_1 + U_2 \sim \chi^2(k_1 + k_2)$ (additivity).

### The $t$ distribution

:::{prf:definition} Student's $t$ distribution
:label: def-t

If $Z \sim \mathcal{N}(0,1)$ and $U \sim \chi^2(\nu)$ are independent, then

$$
T = \frac{Z}{\sqrt{U/\nu}} \sim t(\nu),
$$

the **$t$ distribution with $\nu$ degrees of freedom**.
:::

The $t$ distribution is symmetric about zero with heavier tails than the
standard normal; as $\nu\to\infty$ it converges to $\mathcal{N}(0,1)$.

### The $F$ distribution

:::{prf:definition} $F$ distribution
:label: def-f

If $U_1 \sim \chi^2(\nu_1)$ and $U_2 \sim \chi^2(\nu_2)$ are independent,
then

$$
F = \frac{U_1/\nu_1}{U_2/\nu_2} \sim F(\nu_1, \nu_2),
$$

the **$F$ distribution with $(\nu_1, \nu_2)$ degrees of freedom**.
:::

If $T \sim t(\nu)$, then $T^2 \sim F(1, \nu)$. This is why a partial
$F$-test for a single coefficient is equivalent to a two-sided $t$-test
({doc}`10-anova-and-inference`).

Quantiles are written $\chi^2_{q,\nu}$, $t_{q,\nu}$, and
$F_{q;\nu_1,\nu_2}$ for the $q$-quantile; for example, $t_{1-\alpha/2,\nu}$
is the value exceeded with probability $\alpha/2$.

### Noncentral distributions

When the normal variables do not have mean zero, the same constructions
give noncentral distributions. They describe test statistics when the null
hypothesis is false and are used to compute power
({doc}`11-one-way-anova`).

:::{prf:definition} Noncentral $\chi^2$ distribution
:label: def-noncentral-chi2

If $X_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$, $i = 1,\ldots,k$, are
independent, then

$$
Y = \sum_{i=1}^k \frac{X_i^2}{\sigma_i^2} \sim \chi'^2(k, \delta),
$$

the **noncentral $\chi^2$ distribution** with $k$ degrees of freedom and
**noncentrality parameter**

$$
\delta = \sum_{i=1}^k \frac{\mu_i^2}{\sigma_i^2} .
$$

Its mean is $\E[Y] = k + \delta$, so $\delta = \E[Y] - k$. When
$\delta = 0$ it reduces to the central $\chi^2(k)$ distribution.
:::

:::{prf:definition} Noncentral $F$ distribution
:label: def-noncentral-f

If $X_1 \sim \chi'^2(k_1, \delta)$ and $X_2 \sim \chi^2(k_2)$ are
independent, then

$$
F = \frac{X_1/k_1}{X_2/k_2} \sim F'(k_1, k_2, \delta),
$$

the **noncentral $F$ distribution** with $(k_1, k_2)$ degrees of freedom and
noncentrality parameter $\delta$.
:::

## Quadratic forms in normal vectors

The sums of squares in regression are quadratic forms $\bz\T\mathbf{A}\bz$ in
a normal vector. The following results give their distributions.

:::{prf:theorem} Distribution of a quadratic form
:label: thm-quadratic-form-chi2

Let $\bz \sim \mathcal{N}_m(\boldsymbol\mu, \sigma^2\bI_m)$ and let $\mathbf{A}$
be a symmetric idempotent matrix of rank $r$. Then

$$
\frac{\bz\T\mathbf{A}\bz}{\sigma^2} \sim \chi'^2\!\left(r,\ \frac{\boldsymbol\mu\T\mathbf{A}\boldsymbol\mu}{\sigma^2}\right),
$$

which is the central $\chi^2(r)$ distribution when $\mathbf{A}\boldsymbol\mu = \bzero$.
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

By {prf:ref}`thm-spectral`, a symmetric idempotent matrix of rank $r$ can
be written $\mathbf{A} = \mathbf{Q}_r\mathbf{Q}_r\T$, where
$\mathbf{Q}_r\in\R^{m\times r}$ has orthonormal columns
($\mathbf{Q}_r\T\mathbf{Q}_r = \bI_r$). Let
$\mathbf{w} = \mathbf{Q}_r\T\bz/\sigma$. By
{prf:ref}`thm-mvn-properties`,
$\mathbf{w} \sim \mathcal{N}_r(\mathbf{Q}_r\T\boldsymbol\mu/\sigma,\ \bI_r)$,
so its components are independent with unit variance, and

$$
\begin{aligned}
\frac{\bz\T\mathbf{A}\bz}{\sigma^2} &= \mathbf{w}\T\mathbf{w} \\
&= \sum_{i=1}^r w_i^2 .
\end{aligned}
$$

By {prf:ref}`def-noncentral-chi2` this is noncentral $\chi^2$ with $r$
degrees of freedom and noncentrality
$\lVert\mathbf{Q}_r\T\boldsymbol\mu\rVert^2/\sigma^2 = \boldsymbol\mu\T\mathbf{A}\boldsymbol\mu/\sigma^2$.
:::

:::{prf:theorem} Independence of quadratic and linear forms
:label: thm-quadratic-independence

Let $\bz \sim \mathcal{N}_m(\boldsymbol\mu, \sigma^2\bI_m)$, let $\mathbf{A}$
and $\mathbf{B}$ be symmetric matrices, and let $\mathbf{L}$ be a matrix.

1. If $\mathbf{A}\mathbf{B} = \mathbf{0}$, then $\bz\T\mathbf{A}\bz$ and
   $\bz\T\mathbf{B}\bz$ are independent.
2. If $\mathbf{L}\mathbf{A} = \mathbf{0}$, then $\mathbf{L}\bz$ and
   $\bz\T\mathbf{A}\bz$ are independent.
:::

Both statements follow from property 3 of {prf:ref}`thm-mvn-properties`
when $\mathbf{A}$ and $\mathbf{B}$ are idempotent, because each quadratic form
is a function of $\mathbf{A}\bz$ (or $\mathbf{B}\bz$) and
$\Cov(\mathbf{A}\bz, \mathbf{B}\bz) = \sigma^2\mathbf{A}\mathbf{B}$.

:::{prf:theorem} Cochran's theorem
:label: thm-cochran

Let $\bz \sim \mathcal{N}_m(\bzero, \bI_m)$ and suppose that

$$
\bz\T\bz = \sum_{l=1}^{L} \bz\T\mathbf{A}_l\bz,
$$

where each $\mathbf{A}_l$ is symmetric with rank $r_l$. If
$r_1 + \cdots + r_L = m$, then the quadratic forms $\bz\T\mathbf{A}_l\bz$ are
independent and $\bz\T\mathbf{A}_l\bz \sim \chi^2(r_l)$ for every $l$.
:::

Cochran's theorem is the formal reason that sums of squares in an ANOVA
table are independent $\chi^2$ variables whose degrees of freedom add up.

:::{prf:example} Sample mean and sample variance
:label: ex-sample-variance

Let $Y_1,\ldots,Y_n$ be i.i.d. $\mathcal{N}(\mu,\sigma^2)$ and
$Z_i = (Y_i - \mu)/\sigma$. With $\mathbf{J}_n = \mathbf{1}_n\mathbf{1}_n\T$,

$$
\bz\T\bz = \bz\T\Bigl(\bI_n - \tfrac1n\mathbf{J}_n\Bigr)\bz + \bz\T\Bigl(\tfrac1n\mathbf{J}_n\Bigr)\bz ,
$$

with ranks $n-1$ and $1$. By Cochran's theorem,

$$
\begin{gathered}
\frac{(n-1)S^2}{\sigma^2}
= \frac{1}{\sigma^2}\sum_{i=1}^n (Y_i - \bar{Y})^2 \sim \chi^2(n-1), \\
\frac{n(\bar{Y}-\mu)^2}{\sigma^2} \sim \chi^2(1),
\end{gathered}
$$

and the two are independent, so the sample mean $\bar{Y}$ and the sample
variance $S^2$ are independent.
:::

## Estimation

An **estimator** $\widehat{\theta}$ of a parameter $\theta$ is a function of
the data; its value for a particular data set is an **estimate**.

- The **bias** is $\operatorname{Bias}(\widehat\theta) = \E[\widehat\theta] - \theta$;
  $\widehat\theta$ is **unbiased** if the bias is zero.
- The **standard error** $\mathrm{SE}(\widehat\theta)$ is an estimate of the
  standard deviation of $\widehat\theta$.
- $\widehat\theta_n$ is **consistent** if
  $\widehat\theta_n \to \theta$ in probability as $n \to \infty$.

:::{prf:proposition} Bias–variance decomposition of the mean squared error
:label: prop-mse-decomposition

The **mean squared error (MSE)** of an estimator satisfies

$$
\begin{aligned}
\operatorname{MSE}(\widehat\theta)
&= \E\bigl[(\widehat\theta - \theta)^2\bigr] \\
&= \Var(\widehat\theta) + \bigl[\operatorname{Bias}(\widehat\theta)\bigr]^2 .
\end{aligned}
$$

For a vector estimator $\widehat{\boldsymbol\theta}$,

$$
\begin{aligned}
\E\bigl[\lVert\widehat{\boldsymbol\theta} - \boldsymbol\theta\rVert_2^2\bigr]
&= \tr\bigl(\Cov(\widehat{\boldsymbol\theta})\bigr)
 + \bigl\lVert\E[\widehat{\boldsymbol\theta}] - \boldsymbol\theta\bigr\rVert_2^2 .
\end{aligned}
$$
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

Add and subtract $\E[\widehat\theta]$:

$$
\begin{aligned}
\E\bigl[(\widehat\theta - \theta)^2\bigr]
&= \E\bigl[(\widehat\theta - \E[\widehat\theta])^2\bigr]
 + 2\,\bigl(\E[\widehat\theta] - \theta\bigr)\,\E\bigl[\widehat\theta - \E[\widehat\theta]\bigr]
 + \bigl(\E[\widehat\theta] - \theta\bigr)^2 \\
&= \Var(\widehat\theta) + 0 + \bigl[\operatorname{Bias}(\widehat\theta)\bigr]^2 .
\end{aligned}
$$

The vector case applies this to each component and sums.
:::

An unbiased estimator is not necessarily the best: a biased estimator with
a much smaller variance can have a smaller MSE. Ridge and lasso regression
exploit exactly this trade-off ({doc}`16-ridge-and-lasso`).

### Maximum likelihood

Given independent observations with densities $f(y_i;\boldsymbol\theta)$,
the **likelihood** is $\mathcal{L}(\boldsymbol\theta) = \prod_i f(y_i;\boldsymbol\theta)$
and the **log-likelihood** is $\ell(\boldsymbol\theta) = \ln\mathcal{L}(\boldsymbol\theta)$.
The **maximum likelihood estimator (MLE)** maximizes $\ell$. The **score** is
$\nabla\ell(\boldsymbol\theta)$, and the **Fisher information** is

$$
\mathbf{I}(\boldsymbol\theta) = -\E\left[\frac{\partial^2\ell(\boldsymbol\theta)}{\partial\boldsymbol\theta\,\partial\boldsymbol\theta\T}\right].
$$

:::{prf:theorem} Large-sample properties of the MLE
:label: thm-mle-asymptotic

Under regularity conditions, as $n \to \infty$:

1. the MLE is consistent;
2. it is asymptotically normal,
   $\widehat{\boldsymbol\theta} \overset{\text{approx.}}{\sim} \mathcal{N}\bigl(\boldsymbol\theta,\ \mathbf{I}(\boldsymbol\theta)^{-1}\bigr)$,
   which justifies Wald tests and intervals; and
3. **(Wilks)** for nested models, the likelihood ratio statistic
   $-2\bigl[\ell(\widehat{\boldsymbol\theta}_0) - \ell(\widehat{\boldsymbol\theta})\bigr]$
   is approximately $\chi^2(d)$ under the null hypothesis, where
   $\widehat{\boldsymbol\theta}_0$ is the MLE under the null and $d$ is the
   number of restrictions.
:::

These results underlie the Wald and drop-in-deviance tests of
{doc}`17-generalized-linear-models` and the following chapters.

## Hypothesis testing and confidence intervals

A test of a **null hypothesis** $H_0$ against an **alternative** $H_1$ uses
a test statistic and a rejection region.

| | $H_0$ true | $H_0$ false |
|---|---|---|
| Reject $H_0$ | **type I error** (probability $\alpha$) | correct decision (probability $1-\beta$, the **power**) |
| Do not reject $H_0$ | correct decision | **type II error** (probability $\beta$) |

- The **significance level** $\alpha$ is the maximum probability of a type I
  error that we accept.
- The **$p$-value** is the probability, computed under $H_0$, of a test
  statistic at least as extreme as the one observed. $H_0$ is rejected at
  level $\alpha$ when $p < \alpha$.
- A **$100(1-\alpha)\%$ confidence interval** is a random interval that
  contains the true parameter with probability $1-\alpha$ over repeated
  samples. It consists of the parameter values that a two-sided level-$\alpha$
  test would not reject.

## Limit theorems

:::{prf:theorem} Law of large numbers
:label: thm-lln

If $X_1, X_2,\ldots$ are i.i.d. with finite mean $\mu$, then
$\bar{X}_n = \frac1n\sum_{i=1}^n X_i \to \mu$ in probability (indeed almost
surely) as $n \to \infty$.
:::

:::{prf:theorem} Central limit theorem
:label: thm-clt

If $X_1, X_2,\ldots$ are i.i.d. with mean $\mu$ and finite variance
$\sigma^2 > 0$, then

$$
\frac{\sqrt{n}\,(\bar{X}_n - \mu)}{\sigma} \xrightarrow{d} \mathcal{N}(0,1)
\qquad (n \to \infty).
$$
:::

**Slutsky's theorem** allows unknown quantities to be replaced by consistent
estimates: if $U_n \xrightarrow{d} U$ and $V_n \to c$ in probability, then
$U_n/V_n \xrightarrow{d} U/c$ for $c \ne 0$. Together with the central
limit theorem, it explains why $t$ statistics are approximately normal in
large samples even when the errors are not normal.

## Summary and Common Mistakes
### Summary

- $\chi^2(k)$ is a sum of $k$ squared independent standard normals;
  $t(\nu) = Z/\sqrt{U/\nu}$; $F(\nu_1,\nu_2)$ is a ratio of independent
  scaled $\chi^2$ variables. Nonzero means give noncentral versions.
- A quadratic form $\bz\T\mathbf{A}\bz/\sigma^2$ with symmetric idempotent
  $\mathbf{A}$ of rank $r$ is (noncentral) $\chi^2(r)$; Cochran's theorem gives
  independence of ANOVA sums of squares.
- $\operatorname{MSE} = \Var + \operatorname{Bias}^2$.
- MLEs are consistent and asymptotically normal, and likelihood ratio
  statistics are asymptotically $\chi^2$.

### Common mistakes

- Using the $t$ or $F$ distribution without checking that the numerator and
  denominator are independent.
- Reading a $p$-value as the probability that $H_0$ is true.
- Assuming that an unbiased estimator always has the smallest MSE.

## References

- Casella, G., & Berger, R. L. (2002). *Statistical inference* (2nd ed.). Duxbury.
- Seber, G. A. F., & Lee, A. J. (2003). *Linear regression analysis* (2nd ed.). Wiley. <https://doi.org/10.1002/9780471722199>
- Wikipedia contributors. (n.d.). Noncentral chi-squared distribution. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution>
- Wikipedia contributors. (n.d.). Noncentral F-distribution. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Noncentral_F-distribution>

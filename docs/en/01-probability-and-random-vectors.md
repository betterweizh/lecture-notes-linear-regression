# Probability and Random Vectors

## Overview

Regression models describe the distribution of a response given a set of
predictors, so every later chapter relies on the language of probability.
This chapter reviews random variables, expectation and variance,
conditional expectation, the distributions used in the book, and random
vectors with their mean vectors and covariance matrices. It ends with the
multivariate normal distribution, the basis of exact inference in the
linear model. It assumes calculus and basic matrix notation ({doc}`05-linear-algebra` covers the matrix results used here).

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- compute expectations, variances, and covariances, and apply their
  linearity rules;
- use the laws of total expectation and total variance;
- state the mean and variance of the distributions used in the book;
- compute the mean vector and covariance matrix of a linear transformation
  of a random vector, and the expectation of a quadratic form; and
- state the main properties of the multivariate normal distribution.
:::

## Random variables and distributions

A **random variable** $X$ is a numerical outcome of a random experiment.
Its **cumulative distribution function (CDF)** is
$F(x) = \Prob(X \le x)$, which is non-decreasing and right-continuous with
$F(-\infty) = 0$ and $F(\infty) = 1$.

- A **discrete** random variable takes countably many values and has a
  **probability mass function (PMF)** $f(x) = \Prob(X = x)$.
- A **continuous** random variable has a **probability density function
  (PDF)** $f(x) \ge 0$ with $F(x) = \int_{-\infty}^{x} f(t)\,dt$.

Throughout the book, upper-case letters such as $Y_i$ denote random
variables and lower-case letters such as $y_i$ their observed values when
the distinction matters.

## Expectation and variance

### Expectation

The **expectation** (mean) of $X$ is

$$
\begin{gathered}
\E[X] = \sum_x x\,f(x) \quad\text{(discrete)}, \\
\E[X] = \int_{-\infty}^{\infty} x\,f(x)\,dx \quad\text{(continuous)},
\end{gathered}
$$

and, for a function $g$, $\E[g(X)] = \sum_x g(x)f(x)$ or
$\int g(x)f(x)\,dx$. Expectation is **linear**: for constants $a$ and $b$
and random variables $X$ and $Y$,

$$
\E[aX + bY] = a\,\E[X] + b\,\E[Y],
$$

whether or not $X$ and $Y$ are independent.

### Variance, covariance, and correlation

The **variance** measures the spread of $X$ around its mean:

$$
\begin{aligned}
\Var(X) &= \E\bigl[(X - \E[X])^2\bigr] \\
        &= \E[X^2] - \bigl(\E[X]\bigr)^2 .
\end{aligned}
$$

The **covariance** and **correlation** of $X$ and $Y$ are

$$
\begin{aligned}
\Cov(X,Y) &= \E\bigl[(X-\E[X])(Y-\E[Y])\bigr] \\
          &= \E[XY] - \E[X]\,\E[Y], \\
\operatorname{Corr}(X,Y) &= \frac{\Cov(X,Y)}{\sqrt{\Var(X)\Var(Y)}} \in [-1, 1].
\end{aligned}
$$

For constants $a$, $b$, $c$,

$$
\begin{gathered}
\Var(aX + c) = a^2\Var(X), \\
\Var(aX + bY) = a^2\Var(X) + b^2\Var(Y) + 2ab\,\Cov(X,Y).
\end{gathered}
$$

If $X$ and $Y$ are independent, then $\E[XY] = \E[X]\E[Y]$ and
$\Cov(X,Y) = 0$. The converse is false in general: uncorrelated random
variables need not be independent, but for jointly normal variables they
are ({prf:ref}`thm-mvn-properties`).

### Conditional expectation

The **conditional expectation** $\E[Y\mid X]$ is the mean of $Y$ computed
from the conditional distribution of $Y$ given $X$; it is a function of $X$.
Regression models are models for $\E[Y\mid\bx]$.

:::{prf:theorem} Laws of total expectation and total variance
:label: thm-total-expectation

For random variables $X$ and $Y$ with finite variances,

$$
\begin{gathered}
\E[Y] = \E\bigl[\E[Y\mid X]\bigr], \\
\Var(Y) = \E\bigl[\Var(Y\mid X)\bigr] + \Var\bigl(\E[Y\mid X]\bigr).
\end{gathered}
$$
:::

The second identity splits the variance of $Y$ into the average variation
around the conditional mean and the variation of the conditional mean
itself. It explains, for example, why a Poisson count whose mean varies
randomly is overdispersed ({doc}`19-poisson-and-count-regression`).

## Common distributions

The following distributions appear in the book. For each,
$\E[X]$ and $\Var(X)$ are listed.

### Discrete distributions

| Distribution | PMF $f(x)$ | Support | $\E[X]$ | $\Var(X)$ |
|---|---|---|---|---|
| $\mathrm{Bernoulli}(\pi)$ | $\pi^x(1-\pi)^{1-x}$ | $\{0,1\}$ | $\pi$ | $\pi(1-\pi)$ |
| $\mathrm{Binomial}(m,\pi)$ | $\binom{m}{x}\pi^x(1-\pi)^{m-x}$ | $\{0,\ldots,m\}$ | $m\pi$ | $m\pi(1-\pi)$ |
| $\mathrm{Poisson}(\lambda)$ | $e^{-\lambda}\lambda^x/x!$ | $\{0,1,2,\ldots\}$ | $\lambda$ | $\lambda$ |
| $\mathrm{NegBin}(r,p_s)$ | $\binom{x+r-1}{r-1}(1-p_s)^x p_s^{\,r}$ | $\{0,1,2,\ldots\}$ | $r(1-p_s)/p_s$ | $r(1-p_s)/p_s^2$ |

### Continuous distributions

| Distribution | PDF $f(x)$ | Support | $\E[X]$ | $\Var(X)$ |
|---|---|---|---|---|
| $\mathrm{Uniform}(a,b)$ | $1/(b-a)$ | $[a,b]$ | $(a+b)/2$ | $(b-a)^2/12$ |
| $\mathcal{N}(\mu,\sigma^2)$ | $\dfrac{1}{\sqrt{2\pi}\,\sigma}e^{-(x-\mu)^2/(2\sigma^2)}$ | $\R$ | $\mu$ | $\sigma^2$ |
| $\mathrm{Exponential}(\lambda)$ | $\lambda e^{-\lambda x}$ | $[0,\infty)$ | $1/\lambda$ | $1/\lambda^2$ |
| $\mathrm{Gamma}(\alpha,\theta)$ (shape, scale) | $\dfrac{x^{\alpha-1}e^{-x/\theta}}{\Gamma(\alpha)\theta^\alpha}$ | $(0,\infty)$ | $\alpha\theta$ | $\alpha\theta^2$ |
| $\mathrm{Laplace}(\mu,b)$ | $\dfrac{1}{2b}e^{-\lvert x-\mu\rvert/b}$ | $\R$ | $\mu$ | $2b^2$ |
| $\mathrm{Logistic}(\mu,s)$ | $\dfrac{e^{-(x-\mu)/s}}{s\bigl(1+e^{-(x-\mu)/s}\bigr)^2}$ | $\R$ | $\mu$ | $\pi^2s^2/3$ |
| $\mathrm{Gumbel}(\eta,\mu)$ (location $\eta$; $\mu > 0$, larger $\mu$ means less spread) | $\mu\,e^{-z}e^{-e^{-z}}$, $z = \mu(x-\eta)$ | $\R$ | $\eta + \gamma/\mu$ | $\pi^2/(6\mu^2)$ |

Here $\Gamma(\cdot)$ is the gamma function and $\gamma \approx 0.57721$ is
the Euler–Mascheroni constant. The standard normal distribution
$\mathcal{N}(0,1)$ has CDF $\Phi$ and PDF $\phi$. The logistic CDF is
$F(x) = 1/\bigl(1 + e^{-(x-\mu)/s}\bigr)$; with $\mu = 0$ and $s = 1$ it is the
inverse of the logit function used in {doc}`18-logistic-regression`. The
Gumbel distribution is used in {doc}`21-multinomial-choice-models`, and the
Laplace distribution in {doc}`16-ridge-and-lasso`.

## Random vectors

A **random vector** $\bz = (Z_1,\ldots,Z_m)\T$ is a vector of random
variables. Its **mean vector** and **covariance matrix** are

$$
\E[\bz] = \bigl(\E[Z_1],\ldots,\E[Z_m]\bigr)\T,
$$

$$
\begin{aligned}
\Cov(\bz) &= \E\bigl[(\bz - \E[\bz])(\bz - \E[\bz])\T\bigr] \\
          &= \bigl[\Cov(Z_j, Z_l)\bigr]_{j,l=1}^{m} .
\end{aligned}
$$

The covariance matrix is symmetric and positive semidefinite; its diagonal
entries are the variances $\Var(Z_j)$. It is also written $\Var(\bz)$. The
**cross-covariance** of random vectors $\bz$ and $\mathbf{w}$ is
$\Cov(\bz,\mathbf{w}) = \E\bigl[(\bz-\E[\bz])(\mathbf{w}-\E[\mathbf{w}])\T\bigr]$.

:::{prf:lemma} Linear transformations of random vectors
:label: lem-linear-transform

Let $\mathbf{A}$ and $\mathbf{B}$ be constant (non-random) matrices, $\mathbf{c}$
a constant vector, and $\bz$ a random vector of compatible dimension. Then

$$
\begin{gathered}
\E[\mathbf{A}\bz + \mathbf{c}] = \mathbf{A}\,\E[\bz] + \mathbf{c}, \\
\Cov(\mathbf{A}\bz + \mathbf{c}) = \mathbf{A}\,\Cov(\bz)\,\mathbf{A}\T, \\
\Cov(\mathbf{A}\bz,\ \mathbf{B}\bz) = \mathbf{A}\,\Cov(\bz)\,\mathbf{B}\T .
\end{gathered}
$$
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

The first identity follows from the linearity of expectation applied
component by component. For the second, write $\boldsymbol\mu = \E[\bz]$;
then $\mathbf{A}\bz + \mathbf{c} - \E[\mathbf{A}\bz + \mathbf{c}] = \mathbf{A}(\bz - \boldsymbol\mu)$,
so

$$
\begin{aligned}
\Cov(\mathbf{A}\bz + \mathbf{c})
&= \E\bigl[\mathbf{A}(\bz-\boldsymbol\mu)(\bz-\boldsymbol\mu)\T\mathbf{A}\T\bigr] \\
&= \mathbf{A}\,\E\bigl[(\bz-\boldsymbol\mu)(\bz-\boldsymbol\mu)\T\bigr]\,\mathbf{A}\T \\
&= \mathbf{A}\,\Cov(\bz)\,\mathbf{A}\T .
\end{aligned}
$$

The third identity is proved in the same way.
:::

:::{prf:lemma} Expectation of a quadratic form
:label: lem-quadratic-form-mean

Let $\bz$ be a random vector with mean $\boldsymbol\mu$ and covariance
matrix $\bSigma$, and let $\mathbf{A}$ be a constant symmetric matrix. Then

$$
\E\bigl[\bz\T\mathbf{A}\bz\bigr] = \tr(\mathbf{A}\bSigma) + \boldsymbol\mu\T\mathbf{A}\boldsymbol\mu .
$$
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

A scalar equals its trace, and the trace is linear and cyclic
({prf:ref}`lem-trace-cyclic`). Hence

$$
\begin{aligned}
\E\bigl[\bz\T\mathbf{A}\bz\bigr]
&= \E\bigl[\tr(\mathbf{A}\bz\bz\T)\bigr] \\
 &= \tr\bigl(\mathbf{A}\,\E[\bz\bz\T]\bigr) \\
&= \tr\bigl(\mathbf{A}(\bSigma + \boldsymbol\mu\boldsymbol\mu\T)\bigr) \\
 &= \tr(\mathbf{A}\bSigma) + \boldsymbol\mu\T\mathbf{A}\boldsymbol\mu .
\end{aligned}
$$
:::

This lemma gives a one-line proof that $\E[\mathit{SS}_E] = (n-p)\sigma^2$
in {doc}`09-ols-properties-and-geometry`.

## The multivariate normal distribution

A random vector $\bz\in\R^m$ has a **multivariate normal distribution**
with mean $\boldsymbol\mu$ and positive definite covariance matrix
$\bSigma$, written $\bz \sim \mathcal{N}_m(\boldsymbol\mu, \bSigma)$, if its
density is

$$
f(\bz) = (2\pi)^{-m/2}\,\lvert\bSigma\rvert^{-1/2}
\exp\Bigl[-\tfrac12(\bz - \boldsymbol\mu)\T\bSigma^{-1}(\bz - \boldsymbol\mu)\Bigr],
$$

where $\lvert\bSigma\rvert$ is the determinant of $\bSigma$. The standard
case $\mathcal{N}_m(\bzero, \bI_m)$ is a vector of $m$ independent
$\mathcal{N}(0,1)$ variables.

:::{prf:theorem} Properties of the multivariate normal distribution
:label: thm-mvn-properties

Let $\bz \sim \mathcal{N}_m(\boldsymbol\mu, \bSigma)$.

1. **Linear transformations.** For a constant $q\times m$ matrix $\mathbf{A}$
   and vector $\mathbf{c}$,
   $\mathbf{A}\bz + \mathbf{c} \sim \mathcal{N}_q(\mathbf{A}\boldsymbol\mu + \mathbf{c},\ \mathbf{A}\bSigma\mathbf{A}\T)$.
2. **Marginals.** Every sub-vector of $\bz$ is multivariate normal; in
   particular $Z_j \sim \mathcal{N}(\mu_j, \Sigma_{jj})$.
3. **Independence.** If $\bz$ is partitioned as $(\bz_1\T, \bz_2\T)\T$, then
   $\bz_1$ and $\bz_2$ are independent if and only if
   $\Cov(\bz_1, \bz_2) = \mathbf{0}$. More generally, $\mathbf{A}\bz$ and
   $\mathbf{B}\bz$ are independent if and only if
   $\mathbf{A}\bSigma\mathbf{B}\T = \mathbf{0}$.
:::

Property 3 is special to the normal distribution: zero covariance implies
independence. It is the key step in showing that $\widehat{\bbeta}$ and the
residuals are independent in the normal linear model
({prf:ref}`thm-ols-distribution`).

## Summary and Common Mistakes
### Summary

- Expectation is linear; $\Var(aX + bY) = a^2\Var(X) + b^2\Var(Y) + 2ab\Cov(X,Y)$.
- $\E[Y] = \E[\E[Y\mid X]]$ and
  $\Var(Y) = \E[\Var(Y\mid X)] + \Var(\E[Y\mid X])$.
- $\E[\mathbf{A}\bz] = \mathbf{A}\E[\bz]$,
  $\Cov(\mathbf{A}\bz) = \mathbf{A}\Cov(\bz)\mathbf{A}\T$, and
  $\E[\bz\T\mathbf{A}\bz] = \tr(\mathbf{A}\bSigma) + \boldsymbol\mu\T\mathbf{A}\boldsymbol\mu$.
- Linear transformations of multivariate normal vectors are normal, and for
  jointly normal vectors zero covariance means independence.

### Common mistakes

- Assuming $\Var(X + Y) = \Var(X) + \Var(Y)$ without checking that $X$ and
  $Y$ are uncorrelated.
- Concluding independence from zero covariance for non-normal variables.
- Writing $\Cov(\mathbf{A}\bz) = \mathbf{A}\T\Cov(\bz)\mathbf{A}$; the
  transpose belongs on the right.

## References

- Casella, G., & Berger, R. L. (2002). *Statistical inference* (2nd ed.). Duxbury.
- Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley.
- Seber, G. A. F., & Lee, A. J. (2003). *Linear regression analysis* (2nd ed.). Wiley. <https://doi.org/10.1002/9780471722199>

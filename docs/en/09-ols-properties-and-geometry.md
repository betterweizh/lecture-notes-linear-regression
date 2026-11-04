# Geometry and Statistical Properties of OLS

## Overview

This chapter studies the least squares estimator from two angles. The
geometric view describes fitted values as an orthogonal projection of $\by$
onto the column space of $\bX$. The statistical view derives the mean,
covariance, and sampling distribution of $\widehat{\bbeta}$ and proves that
$\widehat{\sigma}^2 = \mathit{SS}_E/(n-p)$ is unbiased. These results are
the foundation of the tests and intervals in
{doc}`10-anova-and-inference`. It builds on {doc}`08-ols-and-ml-estimation`, the trace and projection matrices ({doc}`05-linear-algebra`), the mean and covariance of random vectors ({doc}`01-probability-and-random-vectors`), and quadratic forms in normal vectors ({doc}`02-sampling-distributions-and-limit-theorems`).

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- define the hat matrix and prove that it is symmetric and idempotent;
- show that residuals are orthogonal to the fitted values and to every
  column of $\bX$;
- derive $\E[\widehat{\bbeta}]$ and $\Cov(\widehat{\bbeta})$; and
- prove that $\E[\mathit{SS}_E] = (n-p)\sigma^2$.
:::

## The hat matrix

Substituting the OLS estimator {eq}`eq-ols` into
$\widehat{\by} = \bX\widehat{\bbeta}$ gives

$$
\begin{aligned}
\widehat{\by} &= \bX(\bX\T\bX)^{-1}\bX\T\by \\
&= \bH\by, \\
\bH &= \bX(\bX\T\bX)^{-1}\bX\T \in \R^{n\times n}.
\end{aligned}
$$ (eq-hat-matrix)

The matrix $\bH$ is called the **hat matrix** because it "puts the hat on
$\by$". It is also called the **projection matrix**, since it projects
$\by$ orthogonally onto the column space of $\bX$. The residual vector is

$$
\begin{aligned}
\be &= \by - \widehat{\by} \\
&= (\bI_n - \bH)\by .
\end{aligned}
$$ (eq-residual-projection)

:::{prf:lemma} Properties of the hat matrix
:label: lem-hat-properties

The matrices $\bH$ and $\bI_n - \bH$ are symmetric and idempotent:

$$
\begin{aligned}
\bH &= \bH\T \\
&= \bH^2, \\
\bI_n - \bH &= (\bI_n - \bH)\T \\
&= (\bI_n - \bH)^2 .
\end{aligned}
$$

Moreover, $\bH\bX = \bX$, $(\bI_n-\bH)\bX = \mathbf{0}$, and
$\tr(\bH) = p$.
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

Because $(\bX\T\bX)^{-1}$ is symmetric,
$\bH\T = \bX\bigl((\bX\T\bX)^{-1}\bigr)\T\bX\T = \bH$. Idempotence follows
from

$$
\begin{aligned}
\bH^2 &= \bX(\bX\T\bX)^{-1}\underbrace{\bX\T\bX(\bX\T\bX)^{-1}}_{\bI_p}\bX\T \\
&= \bH .
\end{aligned}
$$

Then $(\bI_n-\bH)^2 = \bI_n - 2\bH + \bH^2 = \bI_n - \bH$, and symmetry of
$\bI_n-\bH$ is immediate. Next,
$\bH\bX = \bX(\bX\T\bX)^{-1}\bX\T\bX = \bX$, so $(\bI_n-\bH)\bX = \mathbf{0}$.
Finally, by the cyclic property of the trace ({prf:ref}`lem-trace-cyclic`),

$$
\begin{aligned}
\tr(\bH) &= \tr\bigl(\bX(\bX\T\bX)^{-1}\bX\T\bigr) \\
&= \tr\bigl(\bX\T\bX(\bX\T\bX)^{-1}\bigr) \\
&= \tr(\bI_p) \\
&= p .
\end{aligned}
$$
:::

## Orthogonality of residuals

:::{prf:proposition} Orthogonality
:label: prop-orthogonality

The residual vector is orthogonal to every column of $\bX$ and to the vector
of fitted values:

$$
\begin{gathered}
\bX\T\be = \bzero, \\
\widehat{\by}\T\be = 0 .
\end{gathered}
$$

If the model contains an intercept, the residuals also sum to zero:
$\sum_{i=1}^n e_i = 0$.
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

From {eq}`eq-residual-projection` and {prf:ref}`lem-hat-properties`,
$\bX\T\be = \bX\T(\bI_n-\bH)\by = \bigl((\bI_n-\bH)\bX\bigr)\T\by = \bzero$.
For the fitted values,

$$
\begin{aligned}
\widehat{\by}\T\be &= (\bH\by)\T(\bI_n-\bH)\by \\
&= \by\T\bH(\bI_n-\bH)\by \\
&= \by\T(\bH-\bH^2)\by \\
&= 0 .
\end{aligned}
$$

When the first column of $\bX$ is the vector of ones $\mathbf{1}_n$, the
first component of $\bX\T\be = \bzero$ reads $\mathbf{1}_n\T\be = \sum_i e_i = 0$.
:::

Geometrically, $\by$ is decomposed into two orthogonal pieces: the fitted
vector $\widehat{\by}$ in the column space of $\bX$, and the residual vector
$\be$ in its orthogonal complement. The orthogonality is what makes the sum
of squares decomposition in {doc}`10-anova-and-inference` work.

## Mean and covariance of the OLS estimator

The results in this section use the rules for linear transformations of
random vectors ({prf:ref}`lem-linear-transform`).

:::{prf:theorem} Unbiasedness and covariance of $\widehat{\bbeta}$
:label: thm-beta-unbiased

Under assumptions 1–5 of {prf:ref}`asm-classical`,

$$
\begin{gathered}
\E[\widehat{\bbeta}] = \bbeta, \\
\Cov(\widehat{\bbeta}) = \sigma^2(\bX\T\bX)^{-1}.
\end{gathered}
$$
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

Write $\widehat{\bbeta} = \mathbf{A}\by$ with the constant matrix
$\mathbf{A} = (\bX\T\bX)^{-1}\bX\T$, and use $\E[\by] = \bX\bbeta$,
$\Cov(\by) = \sigma^2\bI_n$. By {prf:ref}`lem-linear-transform`,

$$
\begin{aligned}
\E[\widehat{\bbeta}] &= (\bX\T\bX)^{-1}\bX\T\E[\by] \\
&= (\bX\T\bX)^{-1}\bX\T\bX\bbeta \\
&= \bbeta ,
\end{aligned}
$$

and, since $\bX\T\bX$ is symmetric,

$$
\begin{aligned}
\Cov(\widehat{\bbeta}) &= (\bX\T\bX)^{-1}\bX\T\,(\sigma^2\bI_n)\,\bX(\bX\T\bX)^{-1} \\
&= \sigma^2(\bX\T\bX)^{-1} .
\end{aligned}
$$
:::

### Reading the covariance matrix

Write $\mathbf{C} = (\bX\T\bX)^{-1}$ with entries $c_{jl}$, indexed
$j,l = 0,1,\ldots,k$ to match the coefficients. The covariance matrix of
$\widehat{\bbeta}$ collects all variances and covariances of the individual
estimators:

$$
\begin{aligned}
\Cov(\widehat{\bbeta})
&= \E\Bigl[\bigl(\widehat{\bbeta}-\E[\widehat{\bbeta}]\bigr)\bigl(\widehat{\bbeta}-\E[\widehat{\bbeta}]\bigr)\T\Bigr] \\
&= \begin{bmatrix}
\Var(\widehat{\beta}_0) & \cdots & \Cov(\widehat{\beta}_0,\widehat{\beta}_k) \\
\vdots & \ddots & \vdots \\
\Cov(\widehat{\beta}_k,\widehat{\beta}_0) & \cdots & \Var(\widehat{\beta}_k)
\end{bmatrix} \\
&= \sigma^2
\begin{bmatrix}
c_{00} & \cdots & c_{0k} \\
\vdots & \ddots & \vdots \\
c_{k0} & \cdots & c_{kk}
\end{bmatrix}.
\end{aligned}
$$

In particular, $\Var(\widehat{\beta}_j) = \sigma^2 c_{jj}$ and
$\Cov(\widehat{\beta}_j,\widehat{\beta}_l) = \sigma^2 c_{jl}$. The matrix is
symmetric because $\Cov(\widehat{\beta}_j,\widehat{\beta}_l) = \Cov(\widehat{\beta}_l,\widehat{\beta}_j)$.

## Unbiased estimation of the error variance

:::{prf:theorem} Unbiasedness of $\widehat{\sigma}^2$
:label: thm-sigma2-unbiased

Under assumptions 1–5 of {prf:ref}`asm-classical`,
$\E[\mathit{SS}_E] = (n-p)\sigma^2$. Hence
$\widehat{\sigma}^2 = \mathit{SS}_E/(n-p)$ is an unbiased estimator of
$\sigma^2$, whereas the ML estimator $\mathit{SS}_E/n$ is biased downward.
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

By {eq}`eq-residual-projection` and {prf:ref}`lem-hat-properties`,

$$
\begin{aligned}
\mathit{SS}_E &= \be\T\be \\
&= \by\T(\bI_n-\bH)\T(\bI_n-\bH)\by \\
&= \by\T(\bI_n-\bH)\by .
\end{aligned}
$$

A scalar equals its own trace, so by {prf:ref}`lem-trace-cyclic` and the
linearity of expectation and trace,

$$
\begin{aligned}
\E[\mathit{SS}_E]
&= \E\bigl[\tr\bigl(\by\T(\bI_n-\bH)\by\bigr)\bigr] \\
 &= \tr\bigl((\bI_n-\bH)\,\E[\by\by\T]\bigr) \\
&= \tr\bigl((\bI_n-\bH)\bigl(\Cov(\by) + \E[\by]\E[\by]\T\bigr)\bigr) \\
&= \tr\bigl((\bI_n-\bH)\,\sigma^2\bI_n\bigr)
 + \tr\bigl((\bI_n-\bH)\bX\bbeta\bbeta\T\bX\T\bigr) \\
&= \sigma^2\bigl(n - \tr(\bH)\bigr) + 0 \\
 &= \sigma^2(n-p),
\end{aligned}
$$

where the second trace vanishes because $(\bI_n-\bH)\bX = \mathbf{0}$, and
$\tr(\bH) = p$ by {prf:ref}`lem-hat-properties`.
:::

## Sampling distributions under normal errors

Adding the normality assumption gives the exact distributions used for
inference.

:::{prf:theorem} Distribution of the OLS estimators
:label: thm-ols-distribution

Under {prf:ref}`asm-classical` (including normality),

1. $\widehat{\bbeta} \sim \mathcal{N}\bigl(\bbeta,\ \sigma^2(\bX\T\bX)^{-1}\bigr)$,
   and in particular
   $\widehat{\beta}_j \sim \mathcal{N}(\beta_j,\ \sigma^2 c_{jj})$;
2. $\dfrac{(n-p)\,\widehat{\sigma}^2}{\sigma^2} = \dfrac{\mathit{SS}_E}{\sigma^2} \sim \chi^2(n-p)$;
3. the residual vector $\be$ is independent of $\widehat{\by}$ and of
   $\widehat{\bbeta}$; consequently
4. $\mathit{SS}_E$ (equivalently $\widehat{\sigma}^2$) is independent of
   $\widehat{\bbeta}$.
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

1. $\widehat{\bbeta} = (\bX\T\bX)^{-1}\bX\T\by$ is a linear transformation of
   the normal vector $\by$, so it is normal by
   {prf:ref}`thm-mvn-properties`, with the mean and covariance of
   {prf:ref}`thm-beta-unbiased`.
2. Since $(\bI_n-\bH)\bX = \mathbf{0}$, the residual vector is
   $\be = (\bI_n-\bH)\by = (\bI_n-\bH)\bvarepsilon$, so

   $$
   \begin{gathered}
   \frac{\mathit{SS}_E}{\sigma^2} = \frac{\bvarepsilon\T(\bI_n-\bH)\bvarepsilon}{\sigma^2}, \\
   \bvarepsilon \sim \mathcal{N}_n(\bzero, \sigma^2\bI_n).
   \end{gathered}
   $$

   The matrix $\bI_n-\bH$ is symmetric and idempotent with rank
   $\tr(\bI_n-\bH) = n-p$ ({prf:ref}`prop-projection`), so
   {prf:ref}`thm-quadratic-form-chi2` gives $\chi^2(n-p)$.
3. $\be = (\bI_n-\bH)\by$ and $\widehat{\bbeta} = (\bX\T\bX)^{-1}\bX\T\by$
   are jointly normal linear functions of $\by$ with cross-covariance

   $$
   \begin{aligned}
   \Cov\bigl(\be,\ \widehat{\bbeta}\bigr)
   &= (\bI_n-\bH)\,(\sigma^2\bI_n)\,\bX(\bX\T\bX)^{-1} \\
   &= \sigma^2(\bI_n-\bH)\bX(\bX\T\bX)^{-1} \\
   &= \mathbf{0},
   \end{aligned}
   $$

   so they are independent by {prf:ref}`thm-mvn-properties`; and
   $\widehat{\by} = \bX\widehat{\bbeta}$ is a function of $\widehat{\bbeta}$.
4. $\mathit{SS}_E = \be\T\be$ is a function of $\be$, hence independent of
   $\widehat{\bbeta}$.
:::

## Summary and Common Mistakes
### Summary

- $\widehat{\by} = \bH\by$, where the hat matrix $\bH$ is symmetric,
  idempotent, and has trace $p$.
- Residuals are orthogonal to the columns of $\bX$ and to $\widehat{\by}$.
- $\widehat{\bbeta}$ is unbiased with covariance
  $\sigma^2(\bX\T\bX)^{-1}$; $\Var(\widehat{\beta}_j) = \sigma^2 c_{jj}$.
- $\E[\mathit{SS}_E] = (n-p)\sigma^2$, so $\mathit{SS}_E/(n-p)$ is unbiased.
- Under normality, $\widehat{\bbeta}$ is normal,
  $\mathit{SS}_E/\sigma^2 \sim \chi^2(n-p)$, and the two are independent.

### Common mistakes

- Claiming that $\widehat{\by}\T\be = 0$ requires an intercept. It does not;
  only $\sum_i e_i = 0$ requires the column of ones.
- Writing $\E[\by] = \bX\by$; the correct statement is $\E[\by] = \bX\bbeta$.
- Using $\widehat{\sigma}^2$ in place of $\sigma^2$ and still claiming an
  exact normal distribution for $\widehat{\beta}_j$. With estimated variance
  the standardized estimator has a $t$ distribution
  ({doc}`10-anova-and-inference`).

## References

- Seber, G. A. F., & Lee, A. J. (2003). *Linear regression analysis* (2nd ed.). Wiley. <https://doi.org/10.1002/9780471722199>
- He, Z. [何志坚]. (n.d.). *数理统计讲义* [Lecture notes on mathematical statistics]. <https://bookdown.org/hezhijian/book/>
- Shedden, K. (2021). *Statistics 600: Regression analysis* [Lecture notes]. University of Michigan. <https://dept.stat.lsa.umich.edu/~kshedden/stats600/>

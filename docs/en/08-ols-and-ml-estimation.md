# Least Squares and Maximum Likelihood Estimation

## Overview

This chapter derives two estimators of the coefficient vector $\bbeta$:
the ordinary least squares (OLS) estimator, which requires no distributional
assumption, and the maximum likelihood (ML) estimator under normal errors.
The two estimators of $\bbeta$ coincide, but the corresponding estimators of
the error variance $\sigma^2$ differ, which motivates the unbiased estimator
used in all later inference. It builds on {doc}`07-linear-regression-model` and on vector derivatives ({doc}`06-matrix-calculus`).

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- derive the normal equations in scalar and matrix form;
- state the condition under which the OLS estimator is unique;
- derive the ML estimators of $\bbeta$ and $\sigma^2$ under normal errors;
  and
- explain why $\mathit{SS}_E/(n-p)$, not $\mathit{SS}_E/n$, is used to
  estimate $\sigma^2$.
:::

## Ordinary least squares

### Objective function

The **least squares** estimator chooses the coefficients that minimize the
sum of squared deviations between the responses and the linear predictor:

$$
S(\beta_0,\beta_1,\ldots,\beta_k)
= \sum_{i=1}^{n} \bigl(y_i - \beta_0 - \beta_1 x_{i1} - \cdots - \beta_k x_{ik}\bigr)^2 .
$$ (eq-ls-scalar)

The estimator is called **ordinary least squares (OLS)** to distinguish it
from weighted or generalized versions.

### Normal equations in scalar form

Setting the partial derivatives of {eq}`eq-ls-scalar` to zero gives $p$
equations in the $p$ unknowns:

$$
\begin{gathered}
\frac{\partial S}{\partial \beta_0}
= -2\sum_{i=1}^{n}\bigl(y_i-\beta_0-\beta_1x_{i1}-\cdots-\beta_kx_{ik}\bigr) = 0, \\
\frac{\partial S}{\partial \beta_j}
= -2\sum_{i=1}^{n}\bigl(y_i-\beta_0-\beta_1x_{i1}-\cdots-\beta_kx_{ik}\bigr)\,x_{ij} = 0,
\qquad j = 1,\ldots,k .
\end{gathered}
$$

Their solution, denoted $\widehat{\beta}_0,\widehat{\beta}_1,\ldots,\widehat{\beta}_k$,
is the OLS estimate.

### Matrix form

In matrix notation the objective is the squared Euclidean norm of the
deviation vector,

$$
\begin{aligned}
S(\bbeta) &= \lVert \by - \bX\bbeta \rVert_2^2 \\
&= (\by - \bX\bbeta)\T(\by - \bX\bbeta), \\
\widehat{\bbeta} &= \operatorname*{arg\,min}_{\bbeta \in \R^p} S(\bbeta),
\end{aligned}
$$

where $\lVert\mathbf{a}\rVert_2 = (\mathbf{a}\T\mathbf{a})^{1/2}$ is the
Euclidean ($L_2$) norm. Expanding the product,

$$
\begin{aligned}
S(\bbeta)
&= \by\T\by - \by\T\bX\bbeta - \bbeta\T\bX\T\by + \bbeta\T\bX\T\bX\bbeta \\
&= \by\T\by - 2\bbeta\T\bX\T\by + \bbeta\T\bX\T\bX\bbeta ,
\end{aligned}
$$

because $\by\T\bX\bbeta$ is a scalar and therefore equals its transpose
$\bbeta\T\bX\T\by$.

Using the vector-derivative rules of {doc}`06-matrix-calculus`,
$\partial(\mathbf{a}\T\bbeta)/\partial\bbeta = \mathbf{a}$ and
$\partial(\bbeta\T\mathbf{A}\bbeta)/\partial\bbeta = 2\mathbf{A}\bbeta$ for a
symmetric matrix $\mathbf{A}$,

$$
\begin{aligned}
\frac{\partial S}{\partial \bbeta} &= -2\bX\T\by + 2\bX\T\bX\bbeta \\
&= \bzero .
\end{aligned}
$$

This yields the **normal equations**

$$
\bX\T\bX\,\widehat{\bbeta} = \bX\T\by .
$$ (eq-normal-equations)

If $\bX\T\bX$ is nonsingular, the unique solution is

$$
\widehat{\bbeta} = (\bX\T\bX)^{-1}\bX\T\by .
$$ (eq-ols)

The Hessian $\partial^2 S/\partial\bbeta\,\partial\bbeta\T = 2\bX\T\bX$ is
positive definite under the same condition, so {eq}`eq-ols` is the unique
minimizer.

:::{prf:lemma} Invertibility of the Gram matrix
:label: lem-gram-invertible

The $p\times p$ matrix $\bX\T\bX$ is nonsingular if and only if
$\rank(\bX) = p$.
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

For any $\mathbf{a}\in\R^p$,
$\mathbf{a}\T\bX\T\bX\mathbf{a} = \lVert\bX\mathbf{a}\rVert_2^2 \ge 0$, with
equality if and only if $\bX\mathbf{a} = \bzero$. If $\rank(\bX) = p$, the
columns of $\bX$ are linearly independent, so $\bX\mathbf{a} = \bzero$ only
for $\mathbf{a} = \bzero$; hence $\bX\T\bX$ is positive definite and
invertible. Conversely, if $\rank(\bX) < p$, there is $\mathbf{a}\ne\bzero$
with $\bX\mathbf{a} = \bzero$, so $\bX\T\bX\mathbf{a} = \bzero$ and
$\bX\T\bX$ is singular.
:::

This is why {prf:ref}`asm-classical` requires full column rank.

### Simple linear regression

With a single predictor, $y_i = \beta_0 + \beta_1x_i + \varepsilon_i$, the
normal equations have the closed-form solution

$$
\begin{gathered}
\widehat{\beta}_1 = \frac{S_{xy}}{S_{xx}}, \\
\widehat{\beta}_0 = \bar{y} - \widehat{\beta}_1\bar{x},
\end{gathered}
$$

where

$$
\begin{aligned}
S_{xx} &= \sum_{i=1}^{n}(x_i - \bar{x})^2, \\
S_{xy} &= \sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y}) \\
&= \sum_{i=1}^{n}(x_i - \bar{x})\,y_i .
\end{aligned}
$$

## Maximum likelihood estimation

Under the normality assumption,
$\by \sim \mathcal{N}(\bX\bbeta, \sigma^2\bI_n)$. The likelihood function
of $(\bbeta,\sigma^2)$ is the joint density of $\by$:

$$
\begin{aligned}
\mathcal{L}(\bbeta,\sigma^2 \mid \by,\bX)
&= \frac{1}{\sqrt{(2\pi)^n \det(\sigma^2\bI_n)}}
   \exp\!\Bigl[-\tfrac12(\by-\bX\bbeta)\T(\sigma^2\bI_n)^{-1}(\by-\bX\bbeta)\Bigr] \\
&= (2\pi\sigma^2)^{-n/2}
   \exp\!\Bigl[-\frac{1}{2\sigma^2}(\by-\bX\bbeta)\T(\by-\bX\bbeta)\Bigr].
\end{aligned}
$$

The log-likelihood is

$$
\ell(\bbeta,\sigma^2)
= -\frac{n}{2}\ln(2\pi\sigma^2)
  - \frac{1}{2\sigma^2}(\by-\bX\bbeta)\T(\by-\bX\bbeta).
$$ (eq-normal-loglik)

Differentiating with respect to $\bbeta$ and $\sigma^2$:

$$
\begin{gathered}
\frac{\partial\ell}{\partial\bbeta}
= \frac{1}{\sigma^2}\bX\T(\by-\bX\bbeta) = \bzero \qquad \Longrightarrow\quad \bX\T\bX\bbeta = \bX\T\by, \\
\frac{\partial\ell}{\partial\sigma^2}
= -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}(\by-\bX\bbeta)\T(\by-\bX\bbeta) = 0 \qquad \Longrightarrow\quad (\by-\bX\bbeta)\T(\by-\bX\bbeta) = n\sigma^2 .
\end{gathered}
$$

The first condition is again the normal equations
{eq}`eq-normal-equations`, so the ML estimator of $\bbeta$ equals the OLS
estimator. Substituting it into the second condition gives the ML estimator
of the error variance:

$$
\begin{aligned}
\widehat{\bbeta}_{\mathrm{ML}} &= (\bX\T\bX)^{-1}\bX\T\by, \\
\widehat{\sigma}^2_{\mathrm{ML}} &= \frac{1}{n}(\by-\bX\widehat{\bbeta})\T(\by-\bX\widehat{\bbeta}) \\
&= \frac{1}{n}\lVert\by-\widehat{\by}\rVert_2^2 \\
&= \frac{\mathit{SS}_E}{n},
\end{aligned}
$$

where $\widehat{\by} = \bX\widehat{\bbeta}$ is the vector of fitted values.

## Estimating the error variance

The **residual sum of squares** (also called the error sum of squares) is

$$
\begin{aligned}
\mathit{SS}_E &= \sum_{i=1}^{n} e_i^2 \\
&= \lVert\be\rVert_2^2 \\
&= (\by-\bX\widehat{\bbeta})\T(\by-\bX\widehat{\bbeta}), \\
\be &= \by - \widehat{\by}.
\end{aligned}
$$

The ML estimator $\mathit{SS}_E/n$ is biased. The estimator used for
inference divides by the residual degrees of freedom $n-p$:

$$
\begin{aligned}
\widehat{\sigma}^2 &= \frac{\mathit{SS}_E}{n-p} \\
&= \mathit{MS}_E ,
\end{aligned}
$$ (eq-sigma2-unbiased)

which satisfies $\E[\widehat{\sigma}^2] = \sigma^2$
({prf:ref}`thm-sigma2-unbiased` in {doc}`09-ols-properties-and-geometry`).
Intuitively, fitting $p$ coefficients forces the residuals to satisfy $p$
linear constraints ($\bX\T\be = \bzero$), so only $n-p$ of them vary freely.
Throughout the book, $\widehat{\sigma}^2$ without a subscript denotes the
unbiased estimator {eq}`eq-sigma2-unbiased`.

## Summary and Common Mistakes
### Summary

- OLS minimizes $\lVert\by-\bX\bbeta\rVert_2^2$; its solution satisfies the
  normal equations $\bX\T\bX\widehat{\bbeta} = \bX\T\by$.
- The solution $\widehat{\bbeta} = (\bX\T\bX)^{-1}\bX\T\by$ is unique when
  $\rank(\bX) = p$.
- Under normal errors the ML estimator of $\bbeta$ equals the OLS estimator.
- The ML variance estimator $\mathit{SS}_E/n$ is biased;
  $\widehat{\sigma}^2 = \mathit{SS}_E/(n-p)$ is unbiased.

### Common mistakes

- Writing the OLS objective as $\lVert\by-\bX\bbeta\rVert_2$ rather than
  its square. The minimizer is the same, but the derivative used above is
  that of the squared norm.
- Mixing $\by\T\bX$ (a row vector) and $\bX\T\by$ (a column vector) in the
  normal equations.
- Using $\widehat{\sigma}^2_{\mathrm{ML}} = \mathit{SS}_E/n$ in $t$ and $F$
  statistics; the exact distributions require $\mathit{SS}_E/(n-p)$.

## References

- Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley.
- He, Z. [何志坚]. (n.d.). *数理统计讲义* [Lecture notes on mathematical statistics]. <https://bookdown.org/hezhijian/book/>

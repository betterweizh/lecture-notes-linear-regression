# Vector and Matrix Derivatives

## Overview

Least squares, maximum likelihood, and penalized regression all minimize or
maximize a scalar function of a coefficient vector. Setting the derivative
with respect to the whole vector equal to zero gives the estimating
equations in one step. This chapter fixes the layout convention, lists the
derivative rules used in the book, and covers the second-order conditions,
convexity, and subgradients needed for the lasso. It builds on {doc}`05-linear-algebra` and single-variable calculus.

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- compute gradients of linear and quadratic functions of a vector;
- differentiate the least squares objective and derive the normal
  equations;
- use the Hessian to check convexity and to state second-order conditions;
- write a second-order Taylor expansion, the basis of Newton's method; and
- compute the subdifferential of the absolute value function.
:::

## Layout convention

Let $f:\R^p\to\R$ be a scalar function of a vector $\bbeta = (\beta_1,\ldots,\beta_p)\T$.
This book uses the **denominator (gradient) layout**: the derivative of a
scalar with respect to a column vector is a column vector of the same size,

$$
\begin{aligned}
\frac{\partial f}{\partial\bbeta} &= \nabla f(\bbeta) \\
&= \left(\frac{\partial f}{\partial\beta_1},\ \ldots,\ \frac{\partial f}{\partial\beta_p}\right)\T,
\end{aligned}
$$

and the matrix of second derivatives is the $p\times p$ **Hessian**

$$
\frac{\partial^2 f}{\partial\bbeta\,\partial\bbeta\T}
= \left[\frac{\partial^2 f}{\partial\beta_j\,\partial\beta_l}\right]_{j,l=1}^{p},
$$

which is symmetric when the second derivatives are continuous.

## Basic rules

Let $\mathbf{a}\in\R^p$ and $\mathbf{A}\in\R^{p\times p}$ be constant.

| Function $f(\bbeta)$ | Gradient $\partial f/\partial\bbeta$ | Hessian |
|---|---|---|
| $\mathbf{a}\T\bbeta = \bbeta\T\mathbf{a}$ | $\mathbf{a}$ | $\mathbf{0}$ |
| $\bbeta\T\bbeta = \lVert\bbeta\rVert_2^2$ | $2\bbeta$ | $2\bI_p$ |
| $\bbeta\T\mathbf{A}\bbeta$ | $(\mathbf{A} + \mathbf{A}\T)\bbeta$ | $\mathbf{A} + \mathbf{A}\T$ |
| $\bbeta\T\mathbf{A}\bbeta$, $\mathbf{A}$ symmetric | $2\mathbf{A}\bbeta$ | $2\mathbf{A}$ |

For a vector-valued function $\mathbf{g}:\R^p\to\R^m$, the **Jacobian** is
the $m\times p$ matrix $\partial\mathbf{g}/\partial\bbeta\T$ with entries
$\partial g_i/\partial\beta_j$. For example,
$\partial(\bX\bbeta)/\partial\bbeta\T = \bX$. The **chain rule** for a scalar
$f(\mathbf{g}(\bbeta))$ reads

$$
\frac{\partial f}{\partial\bbeta}
= \left(\frac{\partial\mathbf{g}}{\partial\bbeta\T}\right)\T\frac{\partial f}{\partial\mathbf{g}} .
$$

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

For the quadratic form, write
$\bbeta\T\mathbf{A}\bbeta = \sum_{j}\sum_{l}a_{jl}\beta_j\beta_l$.
Differentiating with respect to $\beta_m$ gives

$$
\begin{aligned}
\frac{\partial}{\partial\beta_m}\sum_{j,l}a_{jl}\beta_j\beta_l &= \sum_{l}a_{ml}\beta_l + \sum_{j}a_{jm}\beta_j \\
&= (\mathbf{A}\bbeta)_m + (\mathbf{A}\T\bbeta)_m .
\end{aligned}
$$

The other rules are special cases or follow in the same way.
:::

## The least squares objective

:::{prf:example} Gradient of the residual sum of squares
:label: ex-gradient-rss

Let $S(\bbeta) = \lVert\by - \bX\bbeta\rVert_2^2$. Expanding,

$$
\begin{aligned}
S(\bbeta)
&= (\by - \bX\bbeta)\T(\by - \bX\bbeta) \\
&= \by\T\by - 2\by\T\bX\bbeta + \bbeta\T\bX\T\bX\bbeta .
\end{aligned}
$$

By the table above, with $\mathbf{a} = \bX\T\by$ and the symmetric matrix
$\mathbf{A} = \bX\T\bX$,

$$
\begin{aligned}
\frac{\partial S}{\partial\bbeta} &= -2\bX\T\by + 2\bX\T\bX\bbeta \\
&= -2\bX\T(\by - \bX\bbeta), \\
\frac{\partial^2 S}{\partial\bbeta\,\partial\bbeta\T} &= 2\bX\T\bX .
\end{aligned}
$$

The same result follows from the chain rule with $\mathbf{g}(\bbeta) = \by - \bX\bbeta$
and $f(\mathbf{g}) = \mathbf{g}\T\mathbf{g}$. Setting the gradient to zero
gives the normal equations of {doc}`08-ols-and-ml-estimation`.
:::

## Optimality conditions and convexity

A function $f$ is **convex** if
$f\bigl(t\mathbf{a} + (1-t)\mathbf{b}\bigr) \le t f(\mathbf{a}) + (1-t)f(\mathbf{b})$
for all $\mathbf{a},\mathbf{b}$ and $t\in[0,1]$, and **strictly convex** if the
inequality is strict whenever $\mathbf{a}\ne\mathbf{b}$ and $0<t<1$. A
function is concave if its negative is convex.

:::{prf:theorem} Optimality conditions
:label: thm-optimality

Let $f$ be twice continuously differentiable.

1. If $f$ has a local minimum at $\bbeta^{*}$, then $\nabla f(\bbeta^{*}) = \bzero$.
2. $f$ is convex if and only if its Hessian is positive semidefinite
   everywhere; if the Hessian is positive definite everywhere, $f$ is
   strictly convex.
3. For a convex $f$, every point with $\nabla f(\bbeta^{*}) = \bzero$ is a
   global minimum; for a strictly convex $f$, such a point is the unique
   minimum.
:::

Because the Hessian of the least squares objective is $2\bX\T\bX$, the
objective is convex, and strictly convex when $\bX$ has full column rank;
the OLS estimator is then the unique global minimizer. In the same way,
the canonical-link GLM log-likelihood is strictly concave
({doc}`17-generalized-linear-models`).

### Second-order Taylor expansion

Near a point $\bbeta_0$,

$$
f(\bbeta) \approx f(\bbeta_0) + \nabla f(\bbeta_0)\T(\bbeta - \bbeta_0)
+ \tfrac12(\bbeta - \bbeta_0)\T\mathbf{H}(\bbeta_0)(\bbeta - \bbeta_0),
$$

where $\mathbf{H}$ is the Hessian. Minimizing the right-hand side gives the
Newton–Raphson step
$\bbeta_1 = \bbeta_0 - \mathbf{H}(\bbeta_0)^{-1}\nabla f(\bbeta_0)$, used to
fit GLMs.

## Subgradients and the absolute value

The lasso penalty $\lVert\bbeta\rVert_1 = \sum_j\lvert\beta_j\rvert$ is
convex but not differentiable where some $\beta_j = 0$. For a convex
function $f$, a vector $\mathbf{g}$ is a **subgradient** at $\bbeta$ if
$f(\mathbf{b}) \ge f(\bbeta) + \mathbf{g}\T(\mathbf{b} - \bbeta)$ for all
$\mathbf{b}$. The set of subgradients is the **subdifferential**
$\partial f(\bbeta)$. For the absolute value,

$$
\partial\lvert\beta\rvert =
\begin{cases}
\{1\}, & \beta > 0, \\
[-1, 1], & \beta = 0, \\
\{-1\}, & \beta < 0 .
\end{cases}
$$

:::{prf:theorem} Optimality for convex functions
:label: thm-subgradient-optimality

A convex function $f$ attains its minimum at $\bbeta^{*}$ if and only if
$\bzero \in \partial f(\bbeta^{*})$.
:::

When $f$ is differentiable, the subdifferential contains only the gradient,
and the condition reduces to $\nabla f(\bbeta^{*}) = \bzero$. The condition
is used to derive the soft-thresholding solution of the lasso in
{doc}`16-ridge-and-lasso`.

## Summary and Common Mistakes
### Summary

- $\partial(\mathbf{a}\T\bbeta)/\partial\bbeta = \mathbf{a}$ and
  $\partial(\bbeta\T\mathbf{A}\bbeta)/\partial\bbeta = 2\mathbf{A}\bbeta$
  for symmetric $\mathbf{A}$.
- $\partial\lVert\by-\bX\bbeta\rVert_2^2/\partial\bbeta = -2\bX\T(\by-\bX\bbeta)$
  with Hessian $2\bX\T\bX$.
- A positive semidefinite Hessian means convexity and a positive definite
  Hessian strict convexity; for convex functions a stationary point is a
  global minimum.
- The subdifferential of $\lvert\beta\rvert$ at zero is $[-1,1]$;
  $\bzero\in\partial f(\bbeta^{*})$ characterizes minima of convex functions.

### Common mistakes

- Mixing layouts: in the denominator layout used here,
  $\partial(\mathbf{a}\T\bbeta)/\partial\bbeta = \mathbf{a}$, a column
  vector, not $\mathbf{a}\T$.
- Using $2\mathbf{A}\bbeta$ for a non-symmetric $\mathbf{A}$; the general
  rule is $(\mathbf{A}+\mathbf{A}\T)\bbeta$.
- Treating a zero gradient as a minimum without checking convexity or the
  Hessian.

## References

- Boyd, S., & Vandenberghe, L. (2004). *Convex optimization*. Cambridge University Press. <https://doi.org/10.1017/CBO9780511804441>
- Petersen, K. B., & Pedersen, M. S. (2012). *The matrix cookbook* (Version November 15, 2012). Technical University of Denmark. <https://www2.imm.dtu.dk/pubdb/pubs/3274-full.html>

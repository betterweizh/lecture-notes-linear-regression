# Linear Algebra Review

## Overview

The linear model is written in matrix form, and its main results are
statements about ranks, inverses, traces, projections, and quadratic forms.
This chapter collects the linear algebra used in the book: notation, rank
and invertibility, the trace, eigenvalues and the spectral decomposition,
positive definite matrices, orthogonal projections, the singular value
decomposition, and vector norms. It assumes basic matrix arithmetic: addition, multiplication, and transposition.

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- relate the rank of a matrix to the invertibility of its Gram matrix;
- use the cyclic property of the trace;
- state the spectral decomposition of a symmetric matrix and characterize
  positive (semi)definite matrices;
- recognize orthogonal projection matrices and show that their rank equals
  their trace; and
- use the singular value decomposition and the $L_1$ and $L_2$ norms.
:::

## Notation

Vectors are bold lower-case letters and are column vectors, e.g.
$\bx\in\R^m$; matrices are bold upper-case letters, e.g.
$\mathbf{A}\in\R^{m\times n}$ with entries $a_{ij}$. The transpose is
$\mathbf{A}\T$, the $m\times m$ identity matrix is $\bI_m$, $\mathbf{1}_m$
is the vector of ones, and $\mathbf{0}$ is a zero vector or matrix. A square
matrix is **symmetric** if $\mathbf{A} = \mathbf{A}\T$. For conformable
matrices, $(\mathbf{A}\mathbf{B})\T = \mathbf{B}\T\mathbf{A}\T$ and
$(\mathbf{A}\mathbf{B})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$ when both
inverses exist.

The **inner product** of $\mathbf{a},\mathbf{b}\in\R^m$ is
$\mathbf{a}\T\mathbf{b} = \sum_i a_ib_i$; the vectors are **orthogonal** if
$\mathbf{a}\T\mathbf{b} = 0$.

## Vector norms

| Norm | Definition | Used in |
|---|---|---|
| $L_2$ (Euclidean) | $\lVert\mathbf{a}\rVert_2 = \bigl(\sum_i a_i^2\bigr)^{1/2} = (\mathbf{a}\T\mathbf{a})^{1/2}$ | least squares, ridge regression |
| $L_1$ | $\lVert\mathbf{a}\rVert_1 = \sum_i \lvert a_i\rvert$ | lasso regression, least absolute deviations |
| $L_\infty$ | $\lVert\mathbf{a}\rVert_\infty = \max_i \lvert a_i\rvert$ | |

The squared $L_2$ norm is smooth, whereas the $L_1$ norm is not
differentiable where a component is zero; this difference explains why the
lasso, but not ridge regression, sets coefficients exactly to zero
({doc}`16-ridge-and-lasso`).

## Rank and invertibility

The **column space** of $\mathbf{A}\in\R^{m\times n}$ is
$\mathcal{C}(\mathbf{A}) = \{\mathbf{A}\mathbf{v} : \mathbf{v}\in\R^n\}$. The
**rank** of $\mathbf{A}$ is the dimension of its column space, i.e. the
number of linearly independent columns (equivalently, rows). $\mathbf{A}$
has **full column rank** if $\rank(\mathbf{A}) = n$.

Useful facts:

- $\rank(\mathbf{A}) \le \min(m, n)$ and $\rank(\mathbf{A}\mathbf{B}) \le \min\bigl(\rank(\mathbf{A}), \rank(\mathbf{B})\bigr)$;
- $\rank(\mathbf{A}\T\mathbf{A}) = \rank(\mathbf{A})$;
- a square matrix $\mathbf{A}\in\R^{n\times n}$ is **invertible**
  (nonsingular) if and only if $\rank(\mathbf{A}) = n$, if and only if
  $\det(\mathbf{A}) \ne 0$.

Consequently, the Gram matrix $\bX\T\bX$ of a design matrix
$\bX\in\R^{n\times p}$ is invertible exactly when $\bX$ has full column rank
({prf:ref}`lem-gram-invertible`).

## The trace

The **trace** of a square matrix is the sum of its diagonal entries,
$\tr(\mathbf{A}) = \sum_i a_{ii}$. It is linear,
$\tr(\mathbf{A} + \mathbf{B}) = \tr(\mathbf{A}) + \tr(\mathbf{B})$ and
$\tr(c\mathbf{A}) = c\,\tr(\mathbf{A})$, and satisfies
$\tr(\mathbf{A}\T) = \tr(\mathbf{A})$.

:::{prf:lemma} Cyclic property of the trace
:label: lem-trace-cyclic

If $\mathbf{A}$ is an $m\times n$ matrix and $\mathbf{B}$ is an $n\times m$
matrix, then $\tr(\mathbf{A}\mathbf{B}) = \tr(\mathbf{B}\mathbf{A})$.
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

$$
\begin{aligned}
\tr(\mathbf{A}\mathbf{B}) &= \sum_{i=1}^m\sum_{j=1}^n a_{ij}b_{ji} \\
&= \sum_{j=1}^n\sum_{i=1}^m b_{ji}a_{ij} \\
&= \tr(\mathbf{B}\mathbf{A}).
\end{aligned}
$$
:::

In particular, a scalar quadratic form satisfies
$\mathbf{a}\T\mathbf{M}\mathbf{a} = \tr(\mathbf{M}\mathbf{a}\mathbf{a}\T)$.

## Eigenvalues and the spectral decomposition

A scalar $\lambda$ and a nonzero vector $\mathbf{v}$ form an
**eigenvalue–eigenvector pair** of a square matrix $\mathbf{A}$ if
$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$. The trace equals the sum of the
eigenvalues and the determinant equals their product.

:::{prf:theorem} Spectral decomposition
:label: thm-spectral

A symmetric matrix $\mathbf{A}\in\R^{m\times m}$ has real eigenvalues
$\lambda_1,\ldots,\lambda_m$ and an orthonormal set of eigenvectors
$\mathbf{q}_1,\ldots,\mathbf{q}_m$, so that

$$
\begin{aligned}
\mathbf{A} &= \mathbf{Q}\boldsymbol\Lambda\mathbf{Q}\T \\
&= \sum_{j=1}^m \lambda_j\mathbf{q}_j\mathbf{q}_j\T,
\end{aligned}
$$

where $\mathbf{Q} = [\mathbf{q}_1,\ldots,\mathbf{q}_m]$ is orthogonal
($\mathbf{Q}\T\mathbf{Q} = \mathbf{Q}\mathbf{Q}\T = \bI_m$) and
$\boldsymbol\Lambda = \diag(\lambda_1,\ldots,\lambda_m)$.
:::

## Positive definite matrices and quadratic forms

For a symmetric matrix $\mathbf{A}$, the function
$\mathbf{a}\mapsto\mathbf{a}\T\mathbf{A}\mathbf{a}$ is a **quadratic form**.
$\mathbf{A}$ is

- **positive definite** if $\mathbf{a}\T\mathbf{A}\mathbf{a} > 0$ for all
  $\mathbf{a}\ne\mathbf{0}$, equivalently if all eigenvalues are positive;
- **positive semidefinite** if $\mathbf{a}\T\mathbf{A}\mathbf{a} \ge 0$ for
  all $\mathbf{a}$, equivalently if all eigenvalues are non-negative.

Examples:

- every covariance matrix is positive semidefinite;
- $\bX\T\bX$ is positive semidefinite, because
  $\mathbf{a}\T\bX\T\bX\mathbf{a} = \lVert\bX\mathbf{a}\rVert_2^2 \ge 0$, and
  it is positive definite when $\bX$ has full column rank;
- for any $\lambda > 0$, $\bX\T\bX + \lambda\bI_p$ is positive definite and
  hence invertible, even when $\bX$ does not have full column rank. This is
  the algebraic basis of ridge regression.

## Orthogonal projections and idempotent matrices

A square matrix $\mathbf{P}$ is **idempotent** if $\mathbf{P}^2 = \mathbf{P}$.
A symmetric idempotent matrix is an **orthogonal projection matrix**: it
maps every vector onto the subspace $\mathcal{C}(\mathbf{P})$, and the
remainder $(\bI - \mathbf{P})\mathbf{a}$ is orthogonal to that subspace.

:::{prf:proposition} Properties of projection matrices
:label: prop-projection

Let $\mathbf{P}$ be symmetric and idempotent. Then

1. its eigenvalues are $0$ or $1$;
2. $\rank(\mathbf{P}) = \tr(\mathbf{P})$; and
3. $\bI - \mathbf{P}$ is also symmetric and idempotent, with
   $\rank(\bI - \mathbf{P}) = m - \rank(\mathbf{P})$.
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

If $\mathbf{P}\mathbf{v} = \lambda\mathbf{v}$ with $\mathbf{v}\ne\mathbf{0}$,
then $\lambda\mathbf{v} = \mathbf{P}\mathbf{v} = \mathbf{P}^2\mathbf{v} = \lambda^2\mathbf{v}$,
so $\lambda\in\{0,1\}$. By {prf:ref}`thm-spectral`, the rank is the number of
nonzero eigenvalues, which is the number of ones, which is their sum, the
trace. For part 3,
$(\bI-\mathbf{P})^2 = \bI - 2\mathbf{P} + \mathbf{P}^2 = \bI - \mathbf{P}$,
and its trace is $m - \tr(\mathbf{P})$.
:::

The hat matrix $\bH = \bX(\bX\T\bX)^{-1}\bX\T$ projects onto the column
space of $\bX$ and has rank $\tr(\bH) = p$
({prf:ref}`lem-hat-properties`). The centering matrix
$\bI_n - \frac1n\mathbf{1}_n\mathbf{1}_n\T$ projects onto the vectors whose
entries sum to zero and has rank $n-1$.

## Singular value decomposition

:::{prf:theorem} Singular value decomposition (SVD)
:label: thm-svd

Every matrix $\bX\in\R^{n\times p}$ with $n \ge p$ can be written as

$$
\bX = \mathbf{U}\mathbf{D}\mathbf{V}\T,
$$

where $\mathbf{U}\in\R^{n\times p}$ has orthonormal columns,
$\mathbf{V}\in\R^{p\times p}$ is orthogonal, and
$\mathbf{D} = \diag(d_1,\ldots,d_p)$ with
$d_1 \ge \cdots \ge d_p \ge 0$, the **singular values** of $\bX$.
:::

It follows that $\bX\T\bX = \mathbf{V}\mathbf{D}^2\mathbf{V}\T$, so the
eigenvalues of $\bX\T\bX$ are $d_j^2$. Small singular values signal a nearly
singular $\bX\T\bX$, the algebraic symptom of multicollinearity
({doc}`14-standardization-and-multicollinearity`); the SVD also explains how
ridge regression shrinks the fitted values ({doc}`16-ridge-and-lasso`).

## Summary and Common Mistakes
### Summary

- $\bX\T\bX$ is invertible if and only if $\bX$ has full column rank.
- $\tr(\mathbf{A}\mathbf{B}) = \tr(\mathbf{B}\mathbf{A})$.
- Symmetric matrices have an orthonormal eigenbasis; positive definiteness
  means all eigenvalues are positive.
- Symmetric idempotent matrices are orthogonal projections with eigenvalues
  $0$ and $1$, and rank equal to trace.
- $\bX = \mathbf{U}\mathbf{D}\mathbf{V}\T$ and the eigenvalues of
  $\bX\T\bX$ are the squared singular values.

### Common mistakes

- Writing $(\mathbf{A}\mathbf{B})\T = \mathbf{A}\T\mathbf{B}\T$; the order
  reverses.
- Applying $\tr(\mathbf{A}\mathbf{B}\mathbf{C}) = \tr(\mathbf{B}\mathbf{A}\mathbf{C})$;
  only cyclic permutations are allowed.
- Assuming that $\bX\T\bX$ is invertible whenever $\bX$ has more rows than
  columns; it also needs linearly independent columns.

## References

- Harville, D. A. (1997). *Matrix algebra from a statistician's perspective*. Springer. <https://doi.org/10.1007/b98818>
- Seber, G. A. F., & Lee, A. J. (2003). *Linear regression analysis* (2nd ed.). Wiley. <https://doi.org/10.1002/9780471722199>

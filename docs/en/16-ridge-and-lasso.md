# Ridge and Lasso Regression

## Overview

The OLS estimator is unbiased, but when predictors are strongly correlated
or numerous its variance can be very large ({doc}`14-standardization-and-multicollinearity`).
**Penalized (regularized) regression** minimizes the residual sum of
squares plus a penalty on the size of the coefficients. The penalty
introduces a small bias in exchange for a large reduction in variance, which
can lower the mean squared error of the estimates and of predictions.
**Ridge regression** uses a squared $L_2$ penalty and has a closed-form
solution; the **lasso** uses an $L_1$ penalty and sets some coefficients
exactly to zero, so it also selects variables
({doc}`15-variable-selection`). It builds on {doc}`09-ols-properties-and-geometry`, the bias–variance decomposition ({doc}`02-sampling-distributions-and-limit-theorems`), the SVD and norms ({doc}`05-linear-algebra`), and gradients and subgradients ({doc}`06-matrix-calculus`).

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- write the penalized least squares objective (squared error plus a
  penalty term) and its constrained form;
- derive the ridge estimator and its bias, covariance, and mean squared
  error;
- explain, with the singular value decomposition, how ridge regression
  shrinks the fit, and compute its effective degrees of freedom;
- derive the lasso solution for an orthonormal design (soft thresholding)
  and the coordinate descent update;
- explain why the lasso produces sparse solutions and ridge does not; and
- choose the penalty parameter by cross-validation.
:::

## Notation

| Symbol | Meaning |
|---|---|
| $\bX \in \R^{n\times k}$ | centered and standardized predictor matrix (no column of ones) |
| $\by \in \R^n$ | centered response |
| $\bbeta \in \R^k$ | slope coefficients |
| $\lambda \ge 0$ | penalty (tuning) parameter |
| $\widehat{\bbeta}^{\mathrm{OLS}}$, $\widehat{\bbeta}^{R}_\lambda$, $\widehat{\bbeta}^{L}_\lambda$ | OLS, ridge, and lasso estimators |
| $\bX = \mathbf{U}\mathbf{D}\mathbf{V}\T$ | singular value decomposition, $\mathbf{D} = \diag(d_1,\ldots,d_k)$ |
| $S(z,\gamma) = \sign(z)\,(\lvert z\rvert - \gamma)_+$ | soft-thresholding operator, $(u)_+ = \max(u,0)$ |

## Squared error with a penalty term

### Standardization and the intercept

The penalty treats all coefficients alike, so the predictors must be on a
common scale. Throughout this chapter each predictor is centered and scaled
to unit variance, and the response is centered
({doc}`14-standardization-and-multicollinearity`). The intercept is **not**
penalized: after centering, its estimate is $\widehat{\beta}_0 = \bar{y}$, and
the remaining problem involves only the slopes. Coefficients on the original
scale are recovered by reversing the standardization.

### Penalized least squares

A penalized least squares estimator minimizes the squared error plus a
penalty term:

$$
\widehat{\bbeta}_\lambda
= \operatorname*{arg\,min}_{\bbeta\in\R^k}
\Bigl\{\underbrace{\lVert\by - \bX\bbeta\rVert_2^2}_{\text{squared error}}
 + \underbrace{\lambda\,P(\bbeta)}_{\text{penalty term}}\Bigr\}.
$$ (eq-penalized-ls)

| Method | Penalty $P(\bbeta)$ | Effect |
|---|---|---|
| Ridge | $\lVert\bbeta\rVert_2^2 = \sum_{j=1}^k\beta_j^2$ | shrinks all coefficients toward zero |
| Lasso | $\lVert\bbeta\rVert_1 = \sum_{j=1}^k\lvert\beta_j\rvert$ | shrinks and sets some coefficients exactly to zero |
| Elastic net | $\rho\lVert\bbeta\rVert_1 + (1-\rho)\lVert\bbeta\rVert_2^2$, $0\le\rho\le1$ | combines both |

The tuning parameter $\lambda$ controls the trade-off: $\lambda = 0$ gives
OLS, and $\lambda\to\infty$ forces $\widehat{\bbeta}_\lambda \to \bzero$.

### Constrained form

Problem {eq}`eq-penalized-ls` is equivalent to the constrained problem

$$
\min_{\bbeta}\ \lVert\by - \bX\bbeta\rVert_2^2
\qquad\text{subject to}\qquad
P(\bbeta) \le t,
$$

in the sense that for every $\lambda \ge 0$ there is a budget $t \ge 0$ with
the same solution, and conversely; $\lambda$ is the Lagrange multiplier of
the constraint. The constrained form gives the geometric picture used below
to explain sparsity.

## Ridge regression

### The ridge estimator

:::{prf:theorem} Ridge estimator
:label: thm-ridge

For $\lambda > 0$, the ridge objective
$\lVert\by - \bX\bbeta\rVert_2^2 + \lambda\lVert\bbeta\rVert_2^2$ has the
unique minimizer

$$
\widehat{\bbeta}^{R}_\lambda = (\bX\T\bX + \lambda\bI_k)^{-1}\bX\T\by .
$$ (eq-ridge)
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

By {prf:ref}`ex-gradient-rss` and the rules of {doc}`06-matrix-calculus`,

$$
\begin{aligned}
\frac{\partial}{\partial\bbeta}\Bigl[\lVert\by - \bX\bbeta\rVert_2^2 + \lambda\bbeta\T\bbeta\Bigr]
&= -2\bX\T(\by - \bX\bbeta) + 2\lambda\bbeta \\
&= -2\bX\T\by + 2(\bX\T\bX + \lambda\bI_k)\bbeta .
\end{aligned}
$$

Setting this to zero gives $(\bX\T\bX + \lambda\bI_k)\bbeta = \bX\T\by$. The
Hessian $2(\bX\T\bX + \lambda\bI_k)$ is positive definite for $\lambda > 0$
({doc}`05-linear-algebra`), so the matrix is invertible and the objective is
strictly convex; by {prf:ref}`thm-optimality` the solution is the unique
minimizer.
:::

Unlike OLS, the ridge estimator exists even when $\bX\T\bX$ is singular, for
example under exact collinearity or when $k > n$.

### Bias, covariance, and mean squared error

Write $\mathbf{W}_\lambda = (\bX\T\bX + \lambda\bI_k)^{-1}\bX\T\bX$, so that
$\widehat{\bbeta}^{R}_\lambda = \mathbf{W}_\lambda\widehat{\bbeta}^{\mathrm{OLS}}$
when $\bX$ has full column rank.

:::{prf:proposition} Moments of the ridge estimator
:label: prop-ridge-moments

Under assumptions 1–5 of {prf:ref}`asm-classical`,

$$
\begin{gathered}
\E\bigl[\widehat{\bbeta}^{R}_\lambda\bigr] = (\bX\T\bX + \lambda\bI_k)^{-1}\bX\T\bX\,\bbeta, \\
\operatorname{Bias}\bigl(\widehat{\bbeta}^{R}_\lambda\bigr) = -\lambda\,(\bX\T\bX + \lambda\bI_k)^{-1}\bbeta, \\
\Cov\bigl(\widehat{\bbeta}^{R}_\lambda\bigr) = \sigma^2(\bX\T\bX + \lambda\bI_k)^{-1}\bX\T\bX\,(\bX\T\bX + \lambda\bI_k)^{-1}.
\end{gathered}
$$
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

Let $\mathbf{A}_\lambda = (\bX\T\bX + \lambda\bI_k)^{-1}\bX\T$, so that
$\widehat{\bbeta}^{R}_\lambda = \mathbf{A}_\lambda\by$. By
{prf:ref}`lem-linear-transform`, $\E[\widehat{\bbeta}^{R}_\lambda] = \mathbf{A}_\lambda\bX\bbeta$
and $\Cov(\widehat{\bbeta}^{R}_\lambda) = \sigma^2\mathbf{A}_\lambda\mathbf{A}_\lambda\T$,
which are the first and third results. For the bias,

$$
\begin{aligned}
\E\bigl[\widehat{\bbeta}^{R}_\lambda\bigr] - \bbeta
&= (\bX\T\bX + \lambda\bI_k)^{-1}\bigl[\bX\T\bX - (\bX\T\bX + \lambda\bI_k)\bigr]\bbeta \\
&= -\lambda\,(\bX\T\bX + \lambda\bI_k)^{-1}\bbeta .
\end{aligned}
$$
:::

The ridge estimator is biased toward zero, but its covariance is smaller
than that of OLS: in the eigenbasis of $\bX\T\bX$ each variance is
multiplied by $d_j^4/(d_j^2+\lambda)^2 < 1$. The singular value
decomposition makes the trade-off explicit.

:::{prf:theorem} Mean squared error of ridge regression
:label: thm-ridge-mse

Let $\bX = \mathbf{U}\mathbf{D}\mathbf{V}\T$ with $d_1,\ldots,d_k > 0$ and let
$\boldsymbol\theta = \mathbf{V}\T\bbeta$. Then

$$
\begin{aligned}
\E\bigl[\lVert\widehat{\bbeta}^{R}_\lambda - \bbeta\rVert_2^2\bigr]
&= \underbrace{\sigma^2\sum_{j=1}^k\frac{d_j^2}{(d_j^2+\lambda)^2}}_{\text{total variance}}
 + \underbrace{\lambda^2\sum_{j=1}^k\frac{\theta_j^2}{(d_j^2+\lambda)^2}}_{\text{squared bias}} .
\end{aligned}
$$ (eq-ridge-mse)

At $\lambda = 0$ this equals $\sigma^2\sum_j d_j^{-2}$, the total variance of
OLS. Its derivative with respect to $\lambda$ at $\lambda = 0$ is
$-2\sigma^2\sum_j d_j^{-4} < 0$, so there is always some $\lambda > 0$ for
which ridge regression has a smaller mean squared error than OLS
(Hoerl & Kennard, 1970).
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

Since $\bX\T\bX = \mathbf{V}\mathbf{D}^2\mathbf{V}\T$ and
$\bX\T\bX + \lambda\bI_k = \mathbf{V}(\mathbf{D}^2 + \lambda\bI_k)\mathbf{V}\T$,
{prf:ref}`prop-ridge-moments` gives

$$
\begin{gathered}
\Cov\bigl(\widehat{\bbeta}^{R}_\lambda\bigr)
= \sigma^2\mathbf{V}\,\diag\!\left(\frac{d_j^2}{(d_j^2+\lambda)^2}\right)\mathbf{V}\T, \\
\operatorname{Bias}\bigl(\widehat{\bbeta}^{R}_\lambda\bigr)
= -\lambda\,\mathbf{V}\,\diag\!\left(\frac{1}{d_j^2+\lambda}\right)\boldsymbol\theta .
\end{gathered}
$$

By {prf:ref}`prop-mse-decomposition`, the mean squared error is the trace of
the covariance plus the squared length of the bias; since $\mathbf{V}$ is
orthogonal, both are computed in the rotated coordinates, giving
{eq}`eq-ridge-mse`. Differentiating, the first sum has derivative
$-2\sigma^2\sum_j d_j^2/(d_j^2+\lambda)^3$, which equals
$-2\sigma^2\sum_j d_j^{-4}$ at $\lambda = 0$, while the derivative of the
second sum, $2\lambda\sum_j d_j^2\theta_j^2/(d_j^2+\lambda)^3$, is zero at
$\lambda = 0$.
:::

The variance term decreases and the squared-bias term increases with
$\lambda$; the best $\lambda$ balances the two. The gain is largest when
some $d_j$ are small, i.e. under multicollinearity, because the OLS variance
contains $1/d_j^2$.

### Shrinkage and effective degrees of freedom

The ridge fitted values are

$$
\begin{aligned}
\widehat{\by}^{R}_\lambda
&= \bX(\bX\T\bX + \lambda\bI_k)^{-1}\bX\T\by \\
&= \sum_{j=1}^k \mathbf{u}_j\,\frac{d_j^2}{d_j^2 + \lambda}\,\mathbf{u}_j\T\by ,
\end{aligned}
$$

where $\mathbf{u}_j$ is the $j$-th column of $\mathbf{U}$. OLS projects $\by$
onto each direction $\mathbf{u}_j$ with weight $1$; ridge multiplies the
projection by the **shrinkage factor** $d_j^2/(d_j^2+\lambda) \in (0,1)$,
shrinking most strongly the directions with small $d_j$, which are the
directions in which the data contain little information.

The fitted values are $\widehat{\by}^{R}_\lambda = \bH_\lambda\by$ with
$\bH_\lambda = \bX(\bX\T\bX + \lambda\bI_k)^{-1}\bX\T$. By analogy with
$\tr(\bH) = p$ for OLS, the **effective degrees of freedom** are

$$
\begin{aligned}
\operatorname{df}(\lambda) &= \tr(\bH_\lambda) \\
&= \sum_{j=1}^k\frac{d_j^2}{d_j^2+\lambda},
\end{aligned}
$$

which decreases from $k$ at $\lambda = 0$ toward $0$ as $\lambda\to\infty$.

## Lasso regression

### The lasso estimator

The **lasso** (least absolute shrinkage and selection operator;
Tibshirani, 1996) solves

$$
\widehat{\bbeta}^{L}_\lambda
= \operatorname*{arg\,min}_{\bbeta\in\R^k}
\Bigl\{\lVert\by - \bX\bbeta\rVert_2^2 + \lambda\lVert\bbeta\rVert_1\Bigr\}.
$$ (eq-lasso)

The objective is convex but not differentiable where some $\beta_j = 0$,
so in general there is no closed form. Its solution is characterized by the
subgradient condition of {prf:ref}`thm-subgradient-optimality`: for each $j$,

$$
\begin{gathered}
2\bx_{(j)}\T\bigl(\by - \bX\widehat{\bbeta}^{L}_\lambda\bigr) = \lambda\,\sign\bigl(\widehat{\beta}^{L}_j\bigr) \qquad  \text{if } \widehat{\beta}^{L}_j \ne 0, \\
\bigl\lvert 2\bx_{(j)}\T\bigl(\by - \bX\widehat{\bbeta}^{L}_\lambda\bigr)\bigr\rvert \le \lambda \qquad  \text{if } \widehat{\beta}^{L}_j = 0,
\end{gathered}
$$ (eq-lasso-kkt)

where $\bx_{(j)}$ is the $j$-th column of $\bX$. A predictor is excluded
when its correlation with the current residual is too small to justify the
penalty. In particular, $\widehat{\bbeta}^{L}_\lambda = \bzero$ for all
$\lambda \ge \lambda_{\max} = 2\max_j\lvert\bx_{(j)}\T\by\rvert$.

### Orthonormal design: soft thresholding

When the columns of $\bX$ are orthonormal ($\bX\T\bX = \bI_k$), the three
penalized estimators have explicit forms that show how each treats the OLS
estimate $\widehat{\beta}_j = \bx_{(j)}\T\by$.

:::{prf:theorem} Penalized estimators under an orthonormal design
:label: thm-orthonormal-penalized

If $\bX\T\bX = \bI_k$, then, for $j = 1,\ldots,k$,

$$
\begin{aligned}
\text{ridge:} \quad & \widehat{\beta}^{R}_j = \frac{\widehat{\beta}_j}{1 + \lambda}, \\
\text{lasso:} \quad & \widehat{\beta}^{L}_j = S\bigl(\widehat{\beta}_j,\ \lambda/2\bigr) \\
 &= \sign(\widehat{\beta}_j)\bigl(\lvert\widehat{\beta}_j\rvert - \lambda/2\bigr)_+ , \\
\text{best subset } \bigl(P(\bbeta) = \#\{j:\beta_j\ne0\}\bigr): \quad & \widehat{\beta}^{0}_j = \widehat{\beta}_j\,\mathbb{I}\bigl(\lvert\widehat{\beta}_j\rvert > \sqrt{\lambda}\bigr).
\end{aligned}
$$
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

With $\bX\T\bX = \bI_k$ and $\widehat{\bbeta} = \bX\T\by$,

$$
\begin{aligned}
\lVert\by - \bX\bbeta\rVert_2^2
&= \lVert\by - \bX\widehat{\bbeta}\rVert_2^2 + \lVert\bX(\widehat{\bbeta} - \bbeta)\rVert_2^2 \\
&= \lVert\by - \bX\widehat{\bbeta}\rVert_2^2 + \sum_{j=1}^k(\widehat{\beta}_j - \beta_j)^2 ,
\end{aligned}
$$

because the cross term $2(\by - \bX\widehat{\bbeta})\T\bX(\widehat{\bbeta}-\bbeta)$
vanishes ({prf:ref}`prop-orthogonality`). Each penalized problem therefore
separates into $k$ one-dimensional problems
$\min_{\beta_j}\{(\widehat{\beta}_j - \beta_j)^2 + \lambda P_j(\beta_j)\}$.

- Ridge: setting $-2(\widehat{\beta}_j - \beta_j) + 2\lambda\beta_j = 0$
  gives $\beta_j = \widehat{\beta}_j/(1+\lambda)$.
- Lasso: the condition $0 \in -2(\widehat{\beta}_j - \beta_j) + \lambda\,\partial\lvert\beta_j\rvert$
  gives $\beta_j = \widehat{\beta}_j - \frac{\lambda}{2}\sign(\beta_j)$ when
  $\beta_j \ne 0$, which is consistent only if
  $\lvert\widehat{\beta}_j\rvert > \lambda/2$; otherwise
  $\lvert\widehat{\beta}_j\rvert \le \lambda/2$ and $\beta_j = 0$ satisfies
  $0 \in [-2\widehat{\beta}_j - \lambda,\ -2\widehat{\beta}_j + \lambda]$.
- Best subset: keeping $\beta_j = \widehat{\beta}_j$ costs $\lambda$, and
  setting $\beta_j = 0$ costs $\widehat{\beta}_j^2$; keep the coefficient
  when $\widehat{\beta}_j^2 > \lambda$.
:::

Ridge shrinks every coefficient by the same factor and never sets one to
zero. The lasso subtracts a constant $\lambda/2$ from each absolute
coefficient and sets the small ones to zero (**soft thresholding**). Best
subset selection keeps large coefficients unchanged and removes small ones
(**hard thresholding**).

### Why the lasso produces zeros

In the constrained form, the least squares contours are ellipses centered at
the OLS estimate, and the solution is the first point at which a contour
touches the constraint region. The ridge region
$\{\bbeta : \lVert\bbeta\rVert_2^2 \le t\}$ is a ball with a smooth
boundary, so the contact point generally has all coordinates nonzero. The
lasso region $\{\bbeta : \lVert\bbeta\rVert_1 \le t\}$ is a cross-polytope
(a diamond in two dimensions) with corners on the coordinate axes, so the
contact point often lies at a corner or edge where some coordinates are
exactly zero. The non-differentiability of $\lvert\beta_j\rvert$ at zero,
visible in {eq}`eq-lasso-kkt`, is the analytical counterpart of these
corners.

### Coordinate descent

For a general design, the lasso is usually fitted by **cyclic coordinate
descent** (Friedman, Hastie, & Tibshirani, 2010). Holding all coefficients
except $\beta_j$ fixed, the objective in $\beta_j$ is a one-dimensional
lasso problem with the **partial residual**
$\mathbf{r}_{(j)} = \by - \sum_{l\ne j}\bx_{(l)}\beta_l$.

:::{prf:algorithm} Coordinate descent for the lasso
:label: alg-lasso-cd

1. Standardize the predictors, center the response, and choose a starting
   value, e.g. $\bbeta = \bzero$.
2. Repeat until the coefficients change less than a tolerance; for
   $j = 1,\ldots,k$:

   $$
   \beta_j \leftarrow \frac{S\bigl(\bx_{(j)}\T\mathbf{r}_{(j)},\ \lambda/2\bigr)}{\bx_{(j)}\T\bx_{(j)}} .
   $$
3. Recover the intercept $\widehat{\beta}_0 = \bar{y}$ and transform the
   coefficients back to the original scale.
:::

The update follows from the orthonormal-design result applied to a single
column: minimizing $\lVert\mathbf{r}_{(j)} - \bx_{(j)}\beta_j\rVert_2^2 + \lambda\lvert\beta_j\rvert$
gives $\beta_j\,\bx_{(j)}\T\bx_{(j)} = S(\bx_{(j)}\T\mathbf{r}_{(j)}, \lambda/2)$.
Computing the solution for a decreasing sequence of $\lambda$ values,
starting from $\lambda_{\max}$ and using each solution as the starting value
for the next, gives the **regularization path**.

### Elastic net

When predictors are highly correlated, the lasso tends to select one of
them somewhat arbitrarily. The **elastic net** (Zou & Hastie, 2005) adds a
ridge term,

$$
\min_{\bbeta}\Bigl\{\lVert\by - \bX\bbeta\rVert_2^2 + \lambda_1\lVert\bbeta\rVert_1 + \lambda_2\lVert\bbeta\rVert_2^2\Bigr\},
$$

which keeps the sparsity of the lasso while sharing weight among correlated
predictors.

## Bayesian interpretation

Suppose $\by\mid\bbeta \sim \mathcal{N}(\bX\bbeta, \sigma^2\bI_n)$. The negative
log-posterior under a prior density $\pi(\bbeta)$ is, up to a constant,
$\lVert\by - \bX\bbeta\rVert_2^2/(2\sigma^2) - \ln\pi(\bbeta)$.

- With independent normal priors $\beta_j \sim \mathcal{N}(0,\tau^2)$, the
  posterior mode (and mean) is the ridge estimator with
  $\lambda = \sigma^2/\tau^2$.
- With independent Laplace priors $\beta_j \sim \mathrm{Laplace}(0,b)$, the
  posterior mode is the lasso estimator with $\lambda = 2\sigma^2/b$.

The Laplace prior has a sharp peak at zero, which is the probabilistic
reason the lasso favors exact zeros.

## Choosing the penalty parameter

The penalty parameter $\lambda$ is chosen to minimize an estimate of
prediction error, most often by **$K$-fold cross-validation**:

1. split the data into $K$ folds (commonly $K = 5$ or $10$);
2. for each candidate $\lambda$ and each fold $m$, fit the model on the
   other folds and compute the squared prediction error on fold $m$;
3. average over the folds to obtain $\mathrm{CV}(\lambda)$ and choose the
   $\lambda$ that minimizes it.

The **one-standard-error rule** chooses the largest $\lambda$ whose
$\mathrm{CV}(\lambda)$ is within one standard error of the minimum, giving a
simpler model with comparable accuracy. For ridge regression, leave-one-out
cross-validation has a closed form, analogous to PRESS
({doc}`13-prediction-and-model-assessment`):

$$
\mathrm{CV}_{\mathrm{LOO}}(\lambda)
= \frac1n\sum_{i=1}^n\left(\frac{y_i - \widehat{y}_{i,\lambda}}{1 - h_{ii}(\lambda)}\right)^2,
$$

where $\widehat{y}_{i,\lambda} = \bar{y} + \bx_i\T\widehat{\bbeta}^{R}_\lambda$
is the fitted value with the intercept, and $h_{ii}(\lambda)$ is the $i$-th
diagonal entry of the full ridge hat matrix
$\bH_\lambda + \frac1n\mathbf{1}_n\mathbf{1}_n\T$, the second term accounting
for the estimated intercept.
Information criteria such as AIC and BIC can also be used with
$\operatorname{df}(\lambda)$ in place of $p$; for the lasso, the number of
nonzero coefficients is an unbiased estimate of the degrees of freedom
(Zou, Hastie, & Tibshirani, 2007).

## Inference after penalization

The ridge and lasso estimators are biased, so the $t$- and $F$-tests of
{doc}`10-anova-and-inference` do not apply to them directly. For the lasso,
the selected set of predictors is itself random; computing ordinary
$p$-values for the selected model as if it had been fixed in advance
overstates significance. Penalized regression is therefore used mainly for
prediction and variable screening; formal inference requires specialized
post-selection methods.

## Summary and Common Mistakes
### Summary

- Penalized least squares minimizes $\lVert\by-\bX\bbeta\rVert_2^2 + \lambda P(\bbeta)$
  on standardized predictors with an unpenalized intercept.
- Ridge: $\widehat{\bbeta}^R_\lambda = (\bX\T\bX + \lambda\bI)^{-1}\bX\T\by$,
  biased by $-\lambda(\bX\T\bX + \lambda\bI)^{-1}\bbeta$ but with smaller
  variance; some $\lambda > 0$ always beats OLS in mean squared error.
- Ridge shrinks direction $j$ by $d_j^2/(d_j^2+\lambda)$;
  $\operatorname{df}(\lambda) = \sum_j d_j^2/(d_j^2+\lambda)$.
- Lasso: no closed form; soft thresholding under an orthonormal design;
  coordinate descent in general; produces sparse solutions.
- Choose $\lambda$ by cross-validation, possibly with the one-standard-error
  rule.

### Common mistakes

- Penalizing unstandardized predictors, which makes the result depend on
  the units of measurement.
- Penalizing the intercept.
- Comparing values of $\lambda$ across software without checking how the
  objective is scaled; for example, `scikit-learn` divides the squared error
  by $2n$ in `Lasso` but not in `Ridge` ({doc}`26-penalized-regression-python`).
- Interpreting a coefficient set to zero by the lasso as proof that the
  predictor has no effect.
- Reporting OLS-style standard errors and $p$-values for penalized
  estimates.

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction* (2nd ed.). Springer. <https://doi.org/10.1007/978-0-387-84858-7>
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An introduction to statistical learning: With applications in R*. Springer. <https://doi.org/10.1007/978-1-4614-7138-7>
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software, 33*(1), 1–22. <https://doi.org/10.18637/jss.v033.i01>
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics, 12*(1), 55–67. <https://doi.org/10.1080/00401706.1970.10488634>
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B (Methodological), 58*(1), 267–288. <https://doi.org/10.1111/j.2517-6161.1996.tb02080.x>
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B (Statistical Methodology), 67*(2), 301–320. <https://doi.org/10.1111/j.1467-9868.2005.00503.x>
- Zou, H., Hastie, T., & Tibshirani, R. (2007). On the "degrees of freedom" of the lasso. *The Annals of Statistics, 35*(5), 2173–2192. <https://doi.org/10.1214/009053607000000127>

# Linear Regression: Model and Assumptions

## Overview

Linear regression models the conditional mean of a continuous response as a
linear function of a set of predictors. This chapter defines the model in
scalar and matrix form, fixes the notation used throughout Part II, and
states the assumptions under which the results of later chapters hold. It assumes expectation, variance, and the normal distribution ({doc}`01-probability-and-random-vectors`), and matrix algebra: transpose, inverse, and rank ({doc}`05-linear-algebra`).

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- write the multiple linear regression model in scalar and matrix form;
- state the dimensions of $\by$, $\bX$, $\bbeta$, and $\bvarepsilon$;
- distinguish parameters, estimators, errors, and residuals; and
- list the assumptions of the classical normal linear model and explain
  which results depend on each of them.
:::

## Notation

| Symbol | Type and dimension | Meaning |
|---|---|---|
| $n$ | integer | number of observations |
| $k$ | integer | number of predictors |
| $p$ | integer | number of coefficients, $p = k + 1$ (intercept included) |
| $y_i$ | scalar | response of observation $i$, $i = 1,\ldots,n$ |
| $x_{ij}$ | scalar | value of predictor $j$ for observation $i$, $j = 1,\ldots,k$ |
| $\beta_j$ | scalar | regression coefficient, $j = 0,1,\ldots,k$ |
| $\varepsilon_i$ | random scalar | unobserved error of observation $i$ |
| $\by$ | vector in $\R^n$ | response vector |
| $\bX$ | matrix in $\R^{n\times p}$ | design matrix |
| $\bbeta$ | vector in $\R^p$ | coefficient vector |
| $\bvarepsilon$ | random vector in $\R^n$ | error vector |
| $\widehat{\bbeta}$ | random vector in $\R^p$ | estimator of $\bbeta$ |
| $\widehat{y}_i$, $\widehat{\by}$ | scalar, vector in $\R^n$ | fitted values |
| $e_i$, $\be$ | scalar, vector in $\R^n$ | residuals, $\be = \by - \widehat{\by}$ |

## Model definition

### Scalar form

For observations $i = 1,\ldots,n$, the multiple linear regression model is

$$
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_k x_{ik} + \varepsilon_i .
$$ (eq-mlr-scalar)

The coefficients $\beta_0,\beta_1,\ldots,\beta_k$ are fixed, unknown
constants called **parameters** or **regression coefficients**; $\beta_0$ is
the **intercept**. Once the coefficients are estimated, the **fitted value**
of observation $i$ is

$$
\widehat{y}_i = \widehat{\beta}_0 + \widehat{\beta}_1 x_{i1} + \cdots + \widehat{\beta}_k x_{ik} .
$$

The model has $k$ predictors and $p = k + 1$ coefficients. When the model
has no intercept, $p = k$; this case is treated separately where it matters
(for example in {doc}`10-anova-and-inference`).

### Terminology

Several names are used for the same quantities in different fields.

The variable $y$
: **response**, dependent variable, or endogenous variable.

The variables $x_1,\ldots,x_k$
: **predictors**, independent variables, explanatory variables,
  regressors, covariates, or exogenous variables.

This book uses *response* and *predictor* throughout.

### What is random?

The predictors $x_{ij}$ may have been sampled from some distribution, but
the analysis conditions on their observed values and treats them as fixed
constants. The errors $\varepsilon_i$ are random variables. Consequently:

- each response $y_i$ is a random variable, because it depends on
  $\varepsilon_i$;
- the estimators $\widehat{\beta}_j$ are random variables, because they are
  functions of the responses; and
- the fitted values $\widehat{y}_i$ are random variables that estimate the
  conditional mean $\E[y_i \mid \bx_i]$.

A **parameter** such as $\bbeta$ is a fixed unknown constant, an
**estimator** such as $\widehat{\bbeta}$ is a random vector, and an
**estimate** is the numerical value of an estimator computed from one data
set.

### Matrix form

Stacking the $n$ equations {eq}`eq-mlr-scalar` gives

$$
\by = \bX\bbeta + \bvarepsilon ,
$$ (eq-mlr-matrix)

where

$$
\by = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}_{n\times 1},
\quad
\bX = \begin{bmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1k} \\
1 & x_{21} & x_{22} & \cdots & x_{2k} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \cdots & x_{nk}
\end{bmatrix}_{n\times p},
\quad
\bbeta = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_k \end{bmatrix}_{p\times 1},
\quad
\bvarepsilon = \begin{bmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_n \end{bmatrix}_{n\times 1}.
$$

The matrix $\bX$ is called the **design matrix**. Its first column of ones
corresponds to the intercept. Row $i$ of $\bX$ is
$\bx_i\T = (1, x_{i1}, \ldots, x_{ik})$, so that
$y_i = \bx_i\T\bbeta + \varepsilon_i$.

## Assumptions

The results in Part II rely on the following assumptions. Different results
need different subsets of them; each chapter states which ones it uses.

:::{prf:assumption} Classical normal linear model
:label: asm-classical

1. **Linearity.** The model {eq}`eq-mlr-matrix` holds: $\E[\by] = \bX\bbeta$.
2. **Full rank.** $p < n$ and $\rank(\bX) = p$; that is, no column of
   $\bX$ is a linear combination of the other columns.
3. **Zero mean.** $\E[\varepsilon_i] = 0$ for all $i$.
4. **Homoskedasticity.** $\Var(\varepsilon_i) = \sigma^2$ for all $i$,
   with $\sigma^2 > 0$ unknown.
5. **Uncorrelated errors.** $\Cov(\varepsilon_i, \varepsilon_j) = 0$ for
   $i \neq j$.
6. **Normality.** $\varepsilon_i \sim \mathcal{N}(0,\sigma^2)$.
:::

In matrix form, assumptions 3–5 state that

$$
\begin{gathered}
\E[\bvarepsilon] = \bzero, \\
\Var(\bvarepsilon) = \sigma^2 \bI_n ,
\end{gathered}
$$

and adding assumption 6 gives
$\bvarepsilon \sim \mathcal{N}(\bzero, \sigma^2\bI_n)$. For normally
distributed errors, uncorrelatedness is equivalent to independence, so
assumptions 3–6 together say that the errors are independent and
identically distributed (i.i.d.) $\mathcal{N}(0,\sigma^2)$.

The assumptions play different roles:

- assumption 2 guarantees that the least squares estimator exists and is
  unique ({doc}`08-ols-and-ml-estimation`);
- assumptions 1–5 are enough for $\widehat{\bbeta}$ to be unbiased with
  covariance matrix $\sigma^2(\bX\T\bX)^{-1}$
  ({doc}`09-ols-properties-and-geometry`); and
- assumption 6 is needed for the exact $t$ and $F$ distributions used in
  tests and intervals ({doc}`10-anova-and-inference`).

## Summary and Common Mistakes
### Summary

- The linear model is $\by = \bX\bbeta + \bvarepsilon$ with
  $\bX \in \R^{n\times p}$ and $\bbeta \in \R^p$.
- Predictors are treated as fixed; responses, estimators, fitted values, and
  residuals are random.
- The classical assumptions are linearity, full rank, zero-mean,
  homoskedastic, uncorrelated, and normally distributed errors.

### Common mistakes

- Confusing errors with residuals. The error $\varepsilon_i$ is unobserved;
  the residual $e_i = y_i - \widehat{y}_i$ is computed from the data.
- Counting $p$ inconsistently. In this book $p$ counts every estimated
  coefficient, including the intercept.
- Assuming that the predictors themselves must be normally distributed. The
  normality assumption concerns the errors only.

## References

- Heumann, C., Schomaker, M., & Shalabh. (2016). *Introduction to statistics and data analysis: With exercises, solutions and applications in R*. Springer. <https://doi.org/10.1007/978-3-319-46162-5>
- Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley.
- He, Z. [何志坚]. (n.d.). *数理统计讲义* [Lecture notes on mathematical statistics]. <https://bookdown.org/hezhijian/book/>

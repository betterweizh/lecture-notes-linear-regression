# Linear Regression in R

## Overview

This chapter implements linear regression in R with the base function
`lm()`. It covers model formulas (adding and removing predictors, the
intercept, transformations, categorical predictors, and interactions),
residuals, fitted values and intervals, multicollinearity checks with
`alias()` and `vif()`, and the subset and stepwise selection procedures of
{doc}`15-variable-selection`. It applies the methods of {doc}`07-linear-regression-model`, {doc}`13-prediction-and-model-assessment`, {doc}`14-standardization-and-multicollinearity`, and {doc}`15-variable-selection`. In the snippets, `data` is a data frame with a response `Y` and predictors `X1`, `X2`, …. The base R code and `vif()` were run with R 4.5; the `leaps` calls (`regsubsets()` and its `summary`, `plot`, `coef` and `vcov` methods) were checked against the package source but not executed.

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- specify linear models with R formulas, including transformations,
  factors, and interactions;
- extract residuals and fitted values, and compute confidence and
  prediction intervals;
- detect exact collinearity with `alias()` and compute VIFs with
  `car::vif()`; and
- run best subset and stepwise selection with `regsubsets()` and `step()`.
:::

## Model formulas

### Choosing predictors and the intercept

```r
model <- lm(Y ~ X1 + X2, data = data)          # selected predictors
model <- lm(Y ~ ., data = data)                # all other columns as predictors
model <- lm(Y ~ . - X1 - X2, data = data)      # all except X1 and X2

model <- lm(Y ~ 1 + X1 + X2, data = data)      # with intercept (the default)
model <- lm(Y ~ X1 + X2 - 1, data = data)      # without intercept
model <- lm(Y ~ 0 + X1 + X2, data = data)      # without intercept (same model)
model <- lm(Y ~ 1, data = data)                # intercept only
```

Pass the data frame through the `data` argument rather than calling
`attach(data)`, which makes variables visible globally and can silently pick
up variables with the same name from elsewhere.

### Transformations

Inside a formula, `^`, `*`, `+`, and `-` have special meanings, so arithmetic
on a variable must be protected with `I()`:

```r
# Y on X1, X1^2, X1^3 (a cubic polynomial in X1)
model <- lm(Y ~ X1 + I(X1^2) + I(X1^3), data = data)
model <- lm(Y ~ log(X2), data = data)          # functions need no I()
```

### Categorical predictors

A categorical predictor with $L$ categories is represented by $L-1$
indicator (dummy) variables,

$$
x_{(l)} =
\begin{cases}
1, & \text{if the observation is in category } l, \\
0, & \text{otherwise,}
\end{cases}
\qquad l = 1,\ldots,L-1 .
$$

The category without an indicator is the **reference category**; each
coefficient compares a category with it. R creates the indicators
automatically for factors (and for character variables):

```r
# suppose X2 takes the values 1, 2, 3
data$X2_2 <- as.numeric(data$X2 == 2)
data$X2_3 <- as.numeric(data$X2 == 3)
model1 <- lm(Y ~ X1 + X2_2 + X2_3, data = data)     # manual indicators
model2 <- lm(Y ~ X1 + factor(X2), data = data)      # same model
```

### Interactions

- **Continuous–continuous.** For continuous $x_1$ and $x_2$, the interaction
  is the element-wise (Hadamard) product $x_1 \odot x_2$, which adds one new
  predictor.
- **Categorical–continuous.** For a categorical $x_1$ with $L$ categories and
  a continuous $x_2$, the interaction adds $L-1$ predictors, the products of
  $x_2$ with each indicator of $x_1$; each category then has its own slope
  in $x_2$.
- **Categorical–categorical.** For categorical predictors with $L_1$ and
  $L_2$ categories, the interaction adds $(L_1-1)(L_2-1)$ products of
  indicators.

```r
# four equivalent ways to fit Y on X1, X2 and X1*X2
model1 <- lm(Y ~ X1 * X2, data = data)            # "*": main effects + interaction
model2 <- lm(Y ~ X1 + X2 + X1:X2, data = data)    # ":": interaction only
model3 <- lm(Y ~ X1 + X2 + I(X1 * X2), data = data)
model4 <- lm(Y ~ (X1 + X2)^2, data = data)        # all terms up to order 2
```

`model3` is equivalent to the others only when both variables are numeric;
for factors, use `*` or `:`.

## Results, residuals, and prediction

```r
summary(model)        # coefficients, t-tests, R^2, overall F-test
anova(model)          # sequential (type I) ANOVA table
confint(model, level = 0.95)

resid <- residuals(model)        # e = y - y_hat
fit.val <- fitted(model)         # y_hat
est.val <- predict(model)        # same as fitted(model)
rstandard(model)                 # studentized residuals r_i
hatvalues(model)                 # leverages h_ii
cooks.distance(model)            # Cook's distances D_i
```

`anova(model)` reports the sequential sums of squares of
{prf:ref}`ex-sequential-ss`, which depend on the order of the predictors in
the formula.

The `interval` argument of `predict()` chooses between the confidence
interval for the mean response {eq}`eq-ci-mean` and the prediction interval
for a new observation {eq}`eq-pi`:

```r
pred.ci <- predict(model, newdata = data_test, interval = "confidence", level = 0.95)
pred.pi <- predict(model, newdata = data_test, interval = "prediction", level = 0.95)
```

## Multicollinearity

### Aliases: exact collinearity

**Aliased** predictors are exact linear combinations of other predictors,
i.e. perfectly collinear. For example, if $x_2 = 3x_1 + 1$, then $x_2$ is
aliased. `lm()` reports their coefficients as `NA`, and they must be removed
before VIFs can be computed. `alias()` lists them:

```r
model <- lm(Y ~ ., data = data)
alias(model)
```

Part of an output of `alias()` reads:

```text
Complete :
         (Intercept) V2 V3 V4 ... V13 ... frwy frwynos artnos
art       2          -1  0  0 ...  0  ...  1    0       0
age      86           0  0  0 ... -1  ...  0    0       0
ruralnos  0           0  0  1 ...  0  ...  0   -1      -1
```

Each row expresses an aliased variable in terms of the others:

$$
\begin{gathered}
\mathrm{art} = 2 - \mathrm{V2} + \mathrm{frwy}, \\
\mathrm{age} = 86 - \mathrm{V13}, \\
\mathrm{ruralnos} = \mathrm{V4} - \mathrm{frwynos} - \mathrm{artnos}.
\end{gathered}
$$

### Variance inflation factors

```r
# install.packages("car")
library(car)
vif(model)
```

For models with factors, `vif()` reports generalized VIFs, one per term.

## Variable selection

### Subset selection with `regsubsets()`

Install the package once:

```r
install.packages("leaps")
```

The main arguments of `regsubsets()` are:

| Argument | Type | Meaning |
|---|---|---|
| `nvmax` | integer | maximum size of the subsets to examine |
| `intercept` | logical | whether to include an intercept |
| `method` | character | `"exhaustive"`, `"backward"`, `"forward"`, or `"seqrep"` |
| `force.in` | index or logical vector | predictors that must be kept in every model |
| `nbest` | integer | number of best models to record for each subset size |

The example uses the `Hitters` data from the `ISLR` package, which records
the salaries and performance statistics of baseball players.

```r
library(ISLR)
library(leaps)

names(Hitters)                  # variable names
Hitters <- na.omit(Hitters)     # remove observations with missing values
dim(Hitters)
sum(is.na(Hitters))             # should be 0
```

**Best subset selection.** Fit the best model of each size up to all 19
predictors:

```r
regfit.full <- regsubsets(Salary ~ ., data = Hitters, nvmax = 19)
reg.summary <- summary(regfit.full)
reg.summary
```

The summary contains the selection criteria for the best model of each
size, including $R^2$ (`"rsq"`), $\mathit{SS}_E$ (`"rss"`), adjusted $R^2$
(`"adjr2"`), $C_p$ (`"cp"`), and BIC (`"bic"`):

```r
names(reg.summary)
#> [1] "which"  "rsq"    "rss"    "adjr2"  "cp"     "bic"    "outmat" "obj"
reg.summary$rsq
```

**Plots.** The `plot()` method for `regsubsets` objects shows which
predictors are included in the best model of each size, ordered by the
chosen criterion:

```r
plot(regfit.full, scale = "r2")
plot(regfit.full, scale = "adjr2")
plot(regfit.full, scale = "Cp")
plot(regfit.full, scale = "bic")
```

**Coefficients and covariance matrix:**

```r
coef(regfit.full, 1:3)   # coefficients of the best models with 1 to 3 predictors
vcov(regfit.full, 1)     # covariance matrix of the best one-predictor model
```

**Forward, backward, and sequential-replacement selection:**

```r
regfit.fwd <- regsubsets(Salary ~ ., data = Hitters, nvmax = 19, method = "forward")
summary(regfit.fwd)

regfit.bwd <- regsubsets(Salary ~ ., data = Hitters, nvmax = 19, method = "backward")
summary(regfit.bwd)

regfit.seq <- regsubsets(Salary ~ ., data = Hitters, nvmax = 19, method = "seqrep")
summary(regfit.seq)
```

### Stepwise selection with `step()`

The `step()` function (and `stepAIC()` in the `MASS` package) performs
stepwise selection by AIC: at each step it makes the change that reduces AIC
the most and stops when no change reduces AIC further. For linear models it
uses $n\ln(\mathit{SS}_E/n) + 2p$, the form {eq}`eq-aic-linear`; setting
`k = log(n)` gives BIC instead. Its main arguments are:

| Argument | Type | Meaning |
|---|---|---|
| `direction` | character | `"both"`, `"backward"`, or `"forward"` |
| `scope` | formula or list | the range of models to search |
| `trace` | logical or integer | whether (and how much) to print the selection path |

```r
lm.inter_only <- lm(Salary ~ 1, data = Hitters)   # intercept-only model
lm.all        <- lm(Salary ~ ., data = Hitters)   # full model

# Forward stepwise selection
step.fwd <- step(lm.inter_only, direction = "forward",
                 scope = formula(lm.all), trace = 1)
summary(step.fwd)

# Backward stepwise selection
step.bwd <- step(lm.all, direction = "backward",
                 scope = formula(lm.all), trace = 1)
summary(step.bwd)
```

## Summary and Common Mistakes
### Summary

- `lm(Y ~ ..., data = data)` fits linear models; formulas control
  predictors, the intercept, transformations (`I()`), factors, and
  interactions (`*`, `:`).
- `residuals()`, `fitted()`, `rstandard()`, `hatvalues()`, and
  `cooks.distance()` give the diagnostics of Part II;
  `predict(..., interval = )` gives confidence and prediction intervals.
- `alias()` finds exact collinearity; `car::vif()` computes VIFs.
- `leaps::regsubsets()` performs best subset, forward, backward, and
  sequential-replacement selection; `step()` performs AIC-based stepwise
  selection.

### Common mistakes

- Writing `X1^2` instead of `I(X1^2)` in a formula; without `I()`, `^` means
  "interactions up to order 2", and `X1^2` is just `X1`.
- Treating numeric codes of a categorical variable as a continuous
  predictor; wrap them in `factor()`.
- Using `interval = "confidence"` when a single future value is to be
  predicted.
- Calling `vif()` on a model with aliased (`NA`) coefficients.

## References

- Fox, J., & Weisberg, S. (2019). *An R companion to applied regression* (3rd ed.). Sage.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An introduction to statistical learning: With applications in R*. Springer. <https://doi.org/10.1007/978-1-4614-7138-7>
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *ISLR: Data for an introduction to statistical learning with applications in R* (R package version 1.4) [Computer software]. <https://CRAN.R-project.org/package=ISLR>
- Lumley, T. (2024). *leaps: Regression subset selection* (R package version 3.2) [Computer software]. <https://CRAN.R-project.org/package=leaps>
- R Core Team. (2025). *R: A language and environment for statistical computing* [Computer software]. R Foundation for Statistical Computing. <https://www.R-project.org/>
- Statology. (n.d.). *A complete guide to stepwise regression in R*. <https://www.statology.org/stepwise-regression-r/>

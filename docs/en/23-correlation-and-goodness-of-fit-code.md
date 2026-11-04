# Correlation and Goodness-of-Fit Tests in R and Python

## Overview

This chapter implements the methods of {doc}`03-correlation-analysis` and
{doc}`04-goodness-of-fit-tests` in R and Python. For each statistic it shows the
library function and a short implementation from the definition, so that
the two can be checked against each other. It uses the methods of {doc}`03-correlation-analysis` and {doc}`04-goodness-of-fit-tests`. The R code was run with R 4.5 (base packages only); the Python code was tested with `numpy` 2.5, `pandas` 3.0, and `scipy` 1.18.

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- compute Pearson, Spearman, and Kendall correlations and their tests with
  `cor()`/`cor.test()` in R and `scipy.stats` in Python;
- carry out the $\chi^2$ goodness-of-fit test and the one- and two-sample
  Kolmogorov–Smirnov tests; and
- draw a normal Q–Q plot and compute its coordinates by hand.
:::

## Correlation in R

### Pearson correlation and its tests

`cor()` computes the coefficient {eq}`eq-pearson-r`, and `cor.test()`
carries out the $t$-test {eq}`eq-corr-t` and reports a confidence interval
based on Fisher's $z$-transformation. The data are from Heumann et al.
(2016).

```r
cor.pearson <- function(x, y) {
  sum((x - mean(x)) * (y - mean(y))) /
    sqrt(sum((x - mean(x))^2) * sum((y - mean(y))^2))
}

x1 <- c(10.85, 10.44, 10.50, 10.89, 10.62)
x2 <- c( 7.84,  7.96,  7.81,  7.47,  7.74)
cor(x1, x2, method = "pearson")  # -0.6932787
cor.pearson(x1, x2)              # -0.6932787
# cor(data) returns the correlation matrix of the columns of a data frame

# t-test of H0: rho = 0
cor.t <- function(x, y) {
  n <- length(x)
  r <- cor(x, y)
  r * sqrt(n - 2) / sqrt(1 - r^2)
}
ct <- cor.test(x1, x2, method = "pearson", alternative = "two.sided")
ct$statistic   # t = -1.66622, df = 3, p-value = 0.1943
cor.t(x1, x2)  # -1.66622
ct$conf.int    # 95% CI for rho: (-0.9776, 0.4867)

# Fisher z-test of H0: rho = rho0
cor.z <- function(x, y, rho0 = 0) {
  n <- length(x)
  (atanh(cor(x, y)) - atanh(rho0)) * sqrt(n - 3)
}
cor.z(x1, x2, rho0 = 0.2)   # -1.494787
cor.z(x1, x2, rho0 = 0.7)   # -2.434628
cor.z(x1, x2, rho0 = -0.7)  #  0.018469
```

For many variables at once, `psych::corr.test()` returns the matrix of
correlations together with their $t$-tests.

### Spearman's rank correlation

`cor(method = "spearman")` computes the Pearson correlation of the ranks and
handles ties; the shortcut {eq}`eq-spearman` is exact only without ties.
The tied example is Example 8.5 of Hollander et al. (2014).

```r
spearman.no.ties <- function(x, y) {
  n <- length(x)
  1 - 6 * sum((rank(x) - rank(y))^2) / (n * (n^2 - 1))
}

# no ties: all three agree
cor(x1, x2, method = "spearman")    # -0.7
cor.pearson(rank(x1), rank(x2))     # -0.7
spearman.no.ties(x1, x2)            # -0.7

# ties: the shortcut differs
y1 <- c(7.1, 7.1, 7.2, 8.3, 9.4, 10.5, 11.4)
y2 <- c(2.8, 2.9, 2.8, 2.6, 3.5,  4.6,  5.0)
cor(y1, y2, method = "spearman")    # 0.7
spearman.no.ties(y1, y2)            # 0.7053571

# exact test without ties; S is sum(d_i^2)
cor.test(x1, x2, method = "spearman")  # S = 34, p-value = 0.2333

# large-sample z-test (also usable with ties)
rs <- cor(y1, y2, method = "spearman")
z  <- rs * sqrt(length(y1) - 1)        # 1.714643
2 * pnorm(-abs(z))                     # p-value = 0.0864
```

With ties, `cor.test()` warns that it cannot compute an exact $p$-value.
The package `SuppDists` provides the exact null distribution of $R_S$
(`pSpearman()`).

### Kendall's rank correlation

`cor(method = "kendall")` computes $\tau_B$. The function below counts
concordant and discordant pairs and returns $\tau_A$, $\tau_B$, and the
Kendall score $S$. The untied example is Example 8.1 of Hollander et al.
(2014).

```r
kendall.tau <- function(x, y) {
  n <- length(x)
  nc <- nd <- tx <- ty <- 0
  for (i in 1:(n - 1)) for (j in (i + 1):n) {
    s <- sign(x[i] - x[j]) * sign(y[i] - y[j])
    if (s > 0) nc <- nc + 1 else if (s < 0) nd <- nd + 1
    if (x[i] == x[j]) tx <- tx + 1   # pairs tied in x
    if (y[i] == y[j]) ty <- ty + 1   # pairs tied in y
  }
  n0 <- n * (n - 1) / 2
  c(tau_a = (nc - nd) / n0,
    tau_b = (nc - nd) / sqrt((n0 - tx) * (n0 - ty)),
    S     = nc - nd)
}

k1 <- c(44.4, 45.9, 41.9, 53.3, 44.7, 44.1, 50.7, 45.2, 60.1)
k2 <- c( 2.6,  3.1,  2.5,  5.0,  3.6,  4.0,  5.2,  2.8,  3.8)
cor(k1, k2, method = "kendall")  # 0.4444444
kendall.tau(k1, k2)              # tau_a = tau_b = 0.4444444, S = 16

cor(y1, y2, method = "kendall")  # 0.55 (tau_b)
kendall.tau(y1, y2)              # tau_a = 0.5238095, tau_b = 0.55

# exact test; T is the number of concordant pairs
cor.test(k1, k2, method = "kendall")  # T = 26, p-value = 0.1194

# large-sample z-test
n <- length(k1)
S <- kendall.tau(k1, k2)["S"]
z <- S / sqrt(n * (n - 1) * (2 * n + 5) / 18)  # 1.668115
2 * pnorm(-abs(z))                             # p-value = 0.0953
```

## Goodness-of-fit tests in R

### $\chi^2$ goodness-of-fit test

`chisq.test(x, p)` computes {eq}`eq-chi2-gof` with $k-1$ degrees of
freedom. It does not merge sparse categories (it warns when expected
counts are small) and does not know about estimated parameters; when $r$
parameters were estimated, recompute the $p$-value with $k-1-r$ degrees of
freedom.

```r
# 9:3:3:1 ratio
x <- c(315, 108, 101, 32)
p <- c(9, 3, 3, 1) / 16
chisq.test(x, p = p)   # X-squared = 0.47002, df = 3, p-value = 0.9254

# Poisson(3) with categories 0, 1, ..., 10; rescale.p makes sum(p) = 1
counts <- c(15, 47, 70, 81, 52, 25, 16, 9, 4, 1, 0)
p <- dpois(0:10, lambda = 3)
chisq.test(counts, p = p, rescale.p = TRUE)
# warns that the chi-square approximation may be incorrect:
# the last categories have expected counts below 5 and should be merged
```

### Kolmogorov–Smirnov tests

```r
set.seed(1)
y <- rnorm(30, mean = 0.1, sd = 1.1)

# one-sample test against N(0, 1)
ks.test(y, "pnorm", mean = 0, sd = 1)

# the statistic from its definition
ks.D <- function(y, cdf = pnorm, ...) {
  y <- sort(y)
  n <- length(y)
  F0 <- cdf(y, ...)
  max(pmax((1:n) / n - F0, F0 - (0:(n - 1)) / n))
}
ks.D(y)   # equals ks.test(...)$statistic

# two-sample test
z1 <- rnorm(20, mean = 0.1, sd = 1.1)
z2 <- rnorm(40, mean = -0.1, sd = 1.2)
ks.test(z1, z2)
ks2.D <- function(y1, y2) {
  x <- sort(unique(c(y1, y2)))
  max(abs(ecdf(y1)(x) - ecdf(y2)(x)))
}
ks2.D(z1, z2)   # equals ks.test(z1, z2)$statistic
```

`ks.test()` warns when the data contain ties, because the K–S distribution
assumes a continuous distribution.

### Normal Q–Q plot

```r
qqnorm(y)              # sample quantiles against N(0, 1) quantiles
qqline(y, col = "red") # line through the first and third quartiles

# the same coordinates by hand, with the reference line mean + sd * x
n <- length(y)
x.theo <- qnorm((1:n - 0.5) / n)
plot(x.theo, sort(y), xlab = "Theoretical quantiles", ylab = "Sample quantiles")
abline(a = mean(y), b = sd(y), col = "blue")
```

`qqplot(x, y)` compares two samples directly.

## Correlation and goodness of fit in Python

```python
import numpy as np
import pandas as pd
from scipy import stats

x1 = np.array([10.85, 10.44, 10.50, 10.89, 10.62])
x2 = np.array([7.84, 7.96, 7.81, 7.47, 7.74])

# Pearson correlation, t-test, and Fisher-z confidence interval
r = np.corrcoef(x1, x2)[0, 1]                 # correlation matrix entry
res = stats.pearsonr(x1, x2)                  # r and two-sided p-value
ci = res.confidence_interval(confidence_level=0.95)
print(f"r = {r:.7f}, p = {res.pvalue:.7f}, CI = ({ci.low:.7f}, {ci.high:.7f})")
# r = -0.6932787, p = 0.1942599, CI = (-0.9775936, 0.4866509)
pd.Series(x1).corr(pd.Series(x2), method="pearson")  # DataFrame.corr() for matrices

# Fisher z-test of H0: rho = rho0
def fisher_z(x, y, rho0=0.0):
    n = len(x)
    r = np.corrcoef(x, y)[0, 1]
    z = (np.arctanh(r) - np.arctanh(rho0)) * np.sqrt(n - 3)
    return z, 2 * stats.norm.sf(abs(z))

print(fisher_z(x1, x2, rho0=0.7))   # z = -2.4346, p = 0.0149

# Spearman (Pearson correlation of the ranks, ties handled)
y1 = np.array([7.1, 7.1, 7.2, 8.3, 9.4, 10.5, 11.4])
y2 = np.array([2.8, 2.9, 2.8, 2.6, 3.5, 4.6, 5.0])
print(stats.spearmanr(x1, x2).statistic)   # -0.7
print(stats.spearmanr(y1, y2).statistic)   # 0.7

# Kendall: tau-b by default, tau-c with variant="c"
k1 = np.array([44.4, 45.9, 41.9, 53.3, 44.7, 44.1, 50.7, 45.2, 60.1])
k2 = np.array([2.6, 3.1, 2.5, 5.0, 3.6, 4.0, 5.2, 2.8, 3.8])
print(stats.kendalltau(k1, k2))            # tau = 0.4444, exact p = 0.1194
print(stats.kendalltau(y1, y2).statistic)  # 0.55

# chi-square goodness of fit; ddof = number of estimated parameters
obs = np.array([315, 108, 101, 32])
p = np.array([9, 3, 3, 1]) / 16
print(stats.chisquare(obs, f_exp=obs.sum() * p, ddof=0))  # 0.4700, p = 0.9254

# Kolmogorov-Smirnov tests
rng = np.random.default_rng(1)
y = rng.normal(0.1, 1.1, size=30)
print(stats.kstest(y, stats.norm(loc=0, scale=1).cdf))  # one sample
z1 = rng.normal(0.1, 1.1, size=20)
z2 = rng.normal(-0.1, 1.2, size=40)
print(stats.ks_2samp(z1, z2))                           # two samples

# normal Q-Q plot coordinates with plotting positions (i - 0.5)/n
n = len(y)
x_theo = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
y_sorted = np.sort(y)
# plot y_sorted against x_theo with the line y.mean() + y.std(ddof=1) * x
```

`scipy.stats.probplot()` and `statsmodels.graphics.gofplots.qqplot()` draw
the plot directly; they use slightly different plotting positions, which
does not change its interpretation.

## Summary and Common Mistakes
### Summary

- R: `cor()` and `cor.test()` with `method = "pearson"`, `"spearman"`, or
  `"kendall"`; `chisq.test()`; `ks.test()`; `qqnorm()` and `qqline()`.
- Python: `scipy.stats.pearsonr()`, `spearmanr()`, `kendalltau()`,
  `chisquare()`, `kstest()`, `ks_2samp()`, and `probplot()`.
- Each statistic can be reproduced from its definition in a few lines.

### Common mistakes

- Using the $p$-value of `chisq.test()` after estimating parameters of the
  hypothesized distribution; the degrees of freedom must be reduced by $r$.
- Comparing `cor.test(method = "spearman")$statistic` with $R_S$: R reports
  $S = \sum_i d_i^2$; for Kendall it reports the number of concordant pairs
  $T$.
- Passing a distribution name with positional parameters to
  `scipy.stats.kstest()`; pass a frozen distribution's `cdf`, as above.

## References

- Heumann, C., Schomaker, M., & Shalabh. (2016). *Introduction to statistics and data analysis: With exercises, solutions and applications in R*. Springer. <https://doi.org/10.1007/978-3-319-46162-5>
- Hollander, M., Wolfe, D. A., & Chicken, E. (2014). *Nonparametric statistical methods* (3rd ed.). Wiley.
- Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., … SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods, 17*(3), 261–272. <https://doi.org/10.1038/s41592-019-0686-2>
- R Core Team. (2025). *R: A language and environment for statistical computing* [Computer software]. R Foundation for Statistical Computing. <https://www.R-project.org/>

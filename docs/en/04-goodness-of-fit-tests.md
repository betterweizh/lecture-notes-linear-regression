# Goodness-of-Fit Tests and Q–Q Plots

## Overview

Many methods in this book assume that data, or residuals, follow a
particular distribution: normal errors in the linear model, Poisson counts
in count regression. A **goodness-of-fit** procedure checks whether a
sample is consistent with a hypothesized distribution. This chapter covers
the $\chi^2$ goodness-of-fit test for discrete or grouped data, the
Kolmogorov–Smirnov test for continuous distributions (one- and two-sample),
and the quantile–quantile (Q–Q) plot, the standard graphical check of
normality for regression residuals. Code is in
{doc}`23-correlation-and-goodness-of-fit-code`. It builds on distribution functions ({doc}`01-probability-and-random-vectors`) and on the $\chi^2$ distribution and hypothesis tests ({doc}`02-sampling-distributions-and-limit-theorems`).

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- carry out the $\chi^2$ goodness-of-fit test, including the adjustment of
  the degrees of freedom for estimated parameters;
- carry out the one-sample and two-sample Kolmogorov–Smirnov tests;
- construct a normal Q–Q plot and its reference line; and
- read common departures from normality in a Q–Q plot.
:::

## The $\chi^2$ goodness-of-fit test

### Setting

The observations of a random variable $X$ are grouped into $k$ categories
(the values of a discrete variable, or intervals for a continuous one),
with observed frequencies $n_1,\ldots,n_k$ and $n = \sum_i n_i$. A
hypothesized distribution $F_0$ assigns probabilities $p_1,\ldots,p_k$ to
the categories:

| Category | $1$ | $2$ | $\cdots$ | $k$ | Total |
|---|---|---|---|---|---|
| Observed frequency | $n_1$ | $n_2$ | $\cdots$ | $n_k$ | $n$ |
| Hypothesized probability | $p_1$ | $p_2$ | $\cdots$ | $p_k$ | $1$ |

The hypotheses are

$$
\begin{gathered}
H_0:\ F(x) = F_0(x) \quad\text{versus} \\
H_1:\ F(x) \ne F_0(x).
\end{gathered}
$$

### Test statistic

Under $H_0$ the expected frequency of category $i$ is $np_i$. The
**Pearson $\chi^2$ statistic** compares observed and expected frequencies:

$$
X^2 = \sum_{i=1}^k \frac{(n_i - np_i)^2}{np_i}
\ \overset{\text{approx.}}{\sim}\ \chi^2(k - 1 - r),
$$ (eq-chi2-gof)

where $r$ is the number of parameters of $F_0$ estimated from the data
($r = 0$ when $F_0$ is completely specified; $r = 2$ for a normal
distribution with estimated $\mu$ and $\sigma^2$; $r = 1$ for a Poisson
distribution with estimated $\lambda$). One degree of freedom is lost
because the frequencies sum to $n$.

Large values of $X^2$ indicate a poor fit, so the test is one-sided: reject
$H_0$ at level $\alpha$ when $X^2 \ge \chi^2_{1-\alpha,\,k-1-r}$. Small
values of $X^2$ are consistent with $H_0$.

The $\chi^2$ approximation requires the expected frequencies to be large
enough; a common rule is $np_i \ge 5$ for every category. Categories that
violate it are merged with neighboring ones, and $k$ is reduced
accordingly.

## The Kolmogorov–Smirnov test

### One-sample test

For a continuous hypothesized distribution, grouping loses information.
The **Kolmogorov–Smirnov (K–S) test** compares the **empirical distribution
function**

$$
F_n(x) = \frac{\#\{i : x_i \le x\}}{n},
$$

a step function, with the hypothesized continuous CDF $F_0$, using the
largest vertical distance

$$
D_n = \sup_x\bigl\lvert F_n(x) - F_0(x)\bigr\rvert .
$$

Because $F_n$ jumps at the ordered observations $x_{(1)} \le \cdots \le x_{(n)}$,
the supremum is attained just before or at a jump:

$$
D_n = \max_{1\le i\le n}\max\left\{\frac{i}{n} - F_0(x_{(i)}),\ F_0(x_{(i)}) - \frac{i-1}{n}\right\}.
$$

Under $H_0$ the distribution of $D_n$ does not depend on $F_0$. $H_0$ is
rejected when $D_n$ exceeds a tabulated critical value; for large $n$,
reject when

$$
D_n \ge \frac{K_{1-\alpha}}{\sqrt{n}},
$$

where $K_{1-\alpha}$ is the $(1-\alpha)$-quantile of the Kolmogorov
distribution ($K_{0.95} \approx 1.358$). The test assumes that $F_0$ is fully
specified; when its parameters are estimated from the same data, the
standard critical values are too large (the test is conservative), and
corrected versions such as the Lilliefors test should be used.

### Two-sample test

To test whether two independent samples of sizes $n$ and $m$ come from the
same continuous distribution, compare their empirical distribution
functions $F_n$ and $G_m$:

$$
D_{n,m} = \sup_x\bigl\lvert F_n(x) - G_m(x)\bigr\rvert .
$$

For large samples, reject $H_0$ when
$D_{n,m} \ge K_{1-\alpha}\sqrt{(n+m)/(nm)}$.

### Other tests

The K–S statistic is most sensitive near the center of the distribution.
The **Anderson–Darling test** weights the squared difference between $F_n$
and $F_0$ by $1/\{F_0(1-F_0)\}$ and is more sensitive in the tails; the
**Shapiro–Wilk test** is a powerful test designed specifically for
normality.

## Quantile–quantile plots

### Definition

A **Q–Q plot** compares two distributions by plotting their quantiles
against each other: for each probability $p$, the point
$(x_p, y_p)$ satisfies

$$
\Prob(X \le x_p) = \Prob(Y \le y_p) = p .
$$

If the two distributions are the same, the points lie on the line $y = x$;
if they differ only in location and scale ($Y \overset{d}{=} a + bX$), the
points lie on the straight line $y = a + bx$.

### Normal Q–Q plot

To check whether a sample $y_1,\ldots,y_n$ comes from a normal
distribution, plot the ordered sample $y_{(1)} \le \cdots \le y_{(n)}$
against the standard normal quantiles

$$
x_i = \Phi^{-1}\!\left(\frac{i - 0.5}{n}\right), \qquad i = 1,\ldots,n,
$$

where $(i-0.5)/n$ is a **plotting position**, the empirical probability of
the $i$-th order statistic adjusted so that it lies strictly inside
$(0,1)$. If the sample is normal with mean $\mu$ and standard deviation
$\sigma$, the points lie close to the line

$$
y = \widehat{\mu} + \widehat{\sigma}\,x,
$$

with $\widehat{\mu}$ and $\widehat{\sigma}$ the sample mean and standard
deviation. (R's `qqline()` instead passes the line through the first and
third quartiles, which is less affected by outliers.)

### Reading a Q–Q plot

| Pattern | Interpretation |
|---|---|
| points close to a straight line | consistent with normality |
| S-shape, flatter in the middle, steep at both ends | heavier tails than normal |
| inverted S-shape | lighter tails than normal |
| convex or concave curve | right- or left-skewed distribution |
| a few isolated points far from the line | outliers |

In regression, the Q–Q plot of the residuals checks the normality
assumption ({doc}`13-prediction-and-model-assessment`, {doc}`11-one-way-anova`).
With small samples some wiggle is expected even under normality, so the
plot is judged by its overall shape.

## Summary and Common Mistakes
### Summary

- The $\chi^2$ goodness-of-fit statistic
  $X^2 = \sum_i(n_i - np_i)^2/(np_i)$ is approximately
  $\chi^2(k-1-r)$ under $H_0$; reject for large values.
- The K–S statistic is the largest distance between the empirical and the
  hypothesized distribution functions; the two-sample version compares two
  empirical distribution functions.
- A normal Q–Q plot of the ordered sample against
  $\Phi^{-1}\bigl((i-0.5)/n\bigr)$ is close to a straight line for normal
  data; curvature indicates skewness or non-normal tails.

### Common mistakes

- Using the $\chi^2$ test with expected frequencies below about five
  without merging categories.
- Forgetting to subtract the number of estimated parameters from the
  degrees of freedom of the $\chi^2$ test.
- Applying the K–S test with the standard critical values after estimating
  the parameters of $F_0$ from the same data.
- Treating a non-significant goodness-of-fit test as proof that the
  hypothesized distribution is correct; with small samples the tests have
  little power.

## References

- Heumann, C., Schomaker, M., & Shalabh. (2016). *Introduction to statistics and data analysis: With exercises, solutions and applications in R*. Springer. <https://doi.org/10.1007/978-3-319-46162-5>
- Hollander, M., Wolfe, D. A., & Chicken, E. (2014). *Nonparametric statistical methods* (3rd ed.). Wiley.
- Kanji, G. K. (2006). *100 statistical tests* (3rd ed.). Sage.

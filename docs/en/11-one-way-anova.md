# One-Way Analysis of Variance

## Overview

The **one-way analysis of variance (ANOVA)** compares the means of $a$
groups, for example the yields obtained under $a$ treatments in a designed
experiment. It is a special case of the linear model in which the
predictors are group indicators, and its $F$-test is the overall $F$-test of
{doc}`10-anova-and-inference`. This chapter states the fixed-effects model,
proves the sum of squares decomposition and the distributions of the
sums of squares, and derives the expected mean squares. It then estimates
the treatment means with confidence intervals, compares pairs of means with
Fisher's least significant difference, extends the formulas to unbalanced
designs, checks the model assumptions with residuals, and uses the
noncentral $F$ distribution to compute power and choose the sample size.
Random-effects models and blocking are treated in
{doc}`12-random-effects-and-blocking`. It builds on noncentral distributions and Cochran's theorem ({doc}`02-sampling-distributions-and-limit-theorems`) and on {doc}`10-anova-and-inference`.

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- write the one-way fixed-effects model and express it as a linear
  regression on indicator variables;
- prove $\mathit{SS}_T = \mathit{SS}_{\mathrm{Treatments}} + \mathit{SS}_E$;
- derive the distributions of $\mathit{SS}_{\mathrm{Treatments}}$ and
  $\mathit{SS}_E$ and the expected mean squares;
- carry out the $F$-test for equal treatment means;
- estimate treatment means and differences with confidence intervals, and
  compare pairs of means with Fisher's LSD method;
- check the model assumptions with residual plots; and
- compute the type II error and power of the test with the noncentral $F$
  distribution, and use them to choose the number of replicates.
:::

## Notation

| Symbol | Meaning |
|---|---|
| $a$ | number of treatments (groups) |
| $n$ | number of observations per treatment (balanced design) |
| $N = an$ | total number of observations |
| $y_{ij}$ | observation $j$ under treatment $i$, $i = 1,\ldots,a$, $j = 1,\ldots,n$ |
| $\mu$, $\tau_i$ | overall mean and effect of treatment $i$ |
| $y_{i\cdot} = \sum_j y_{ij}$, $\bar{y}_{i\cdot} = y_{i\cdot}/n$ | total and mean of treatment $i$ |
| $y_{\cdot\cdot} = \sum_i\sum_j y_{ij}$, $\bar{y}_{\cdot\cdot} = y_{\cdot\cdot}/N$ | grand total and grand mean |

A dot in a subscript indicates summation over that index.

## Single-factor experiments

In a **single-factor experiment**, one factor is varied over $a$ levels,
called **treatments**, and each treatment is applied to $n$ experimental
units; the observations under one treatment are its **replicates**. The runs
are carried out in **random order**, so that uncontrolled influences
(drift in equipment, changes in operators or raw material) are spread
evenly over the treatments rather than confounded with them. An experiment
moves through a conjecture (the hypothesis that motivates it), the
experiment itself, the statistical analysis, and a conclusion, which often
leads to a revised conjecture and a new experiment.

The data are arranged with one row per treatment:

| Treatment | Observations | Total | Mean |
|---|---|---|---|
| $1$ | $y_{11},\ y_{12},\ \ldots,\ y_{1n}$ | $y_{1\cdot}$ | $\bar{y}_{1\cdot}$ |
| $2$ | $y_{21},\ y_{22},\ \ldots,\ y_{2n}$ | $y_{2\cdot}$ | $\bar{y}_{2\cdot}$ |
| $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ |
| $a$ | $y_{a1},\ y_{a2},\ \ldots,\ y_{an}$ | $y_{a\cdot}$ | $\bar{y}_{a\cdot}$ |
| | | $y_{\cdot\cdot}$ | $\bar{y}_{\cdot\cdot}$ |

The treatments are **fixed** when the experimenter has chosen these
specific $a$ levels; the conclusions then apply only to them, and the model
parameters $\mu$, $	au_i$, and $\sigma^2$ are of direct interest. They are
**random** when the $a$ levels are a random sample from a larger population
of levels; the interest is then in the variability of the whole population
({doc}`12-random-effects-and-blocking`). This chapter treats the fixed case.

## The fixed-effects model

:::{prf:assumption} One-way fixed-effects model
:label: asm-one-way

$$
y_{ij} = \mu + \tau_i + \varepsilon_{ij},
\qquad i = 1,\ldots,a,\quad j = 1,\ldots,n,
$$

where the treatment effects are fixed constants satisfying
$\sum_{i=1}^a\tau_i = 0$, and the errors $\varepsilon_{ij}$ are i.i.d.
$\mathcal{N}(0,\sigma^2)$.
:::

The **treatment means** are $\mu_i = \mu + \tau_i$. The constraint
$\sum_i\tau_i = 0$ makes $\mu$ the average of the treatment means and the
parameters identifiable. The question of interest is whether the treatments
differ:

$$
\begin{gathered}
H_0 :\ \tau_1 = \tau_2 = \cdots = \tau_a = 0 \quad\text{versus} \\
H_1 :\ \tau_i \ne 0 \text{ for at least one } i,
\end{gathered}
$$

equivalently $H_0:\ \mu_1 = \mu_2 = \cdots = \mu_a$.

### The model as a linear regression

Define $a-1$ indicator (dummy) variables $x_{(l)} = 1$ if the observation
belongs to treatment $l$ and $0$ otherwise, $l = 1,\ldots,a-1$, with
treatment $a$ as the reference category. Then

$$
y = \beta_0 + \beta_1x_{(1)} + \cdots + \beta_{a-1}x_{(a-1)} + \varepsilon,
$$

with $\beta_0 = \mu_a$ and $\beta_l = \mu_l - \mu_a$. All treatment means are
equal exactly when $\beta_1 = \cdots = \beta_{a-1} = 0$, so the one-way ANOVA
$F$-test below is the overall $F$-test {eq}`eq-overall-f` of this regression,
with $p = a$ coefficients.

## Sum of squares decomposition

Define

$$
\begin{gathered}
\mathit{SS}_T = \sum_{i=1}^a\sum_{j=1}^n \bigl(y_{ij} - \bar{y}_{\cdot\cdot}\bigr)^2, \\
\mathit{SS}_{\mathrm{Treatments}} = n\sum_{i=1}^a \bigl(\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot}\bigr)^2, \\
\mathit{SS}_E = \sum_{i=1}^a\sum_{j=1}^n \bigl(y_{ij} - \bar{y}_{i\cdot}\bigr)^2 .
\end{gathered}
$$

$\mathit{SS}_{\mathrm{Treatments}}$ measures the variation **between**
treatments and $\mathit{SS}_E$ the variation **within** treatments.

:::{prf:theorem} One-way ANOVA identity
:label: thm-one-way-decomposition

$$
\mathit{SS}_T = \mathit{SS}_{\mathrm{Treatments}} + \mathit{SS}_E .
$$
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

Write $y_{ij} - \bar{y}_{\cdot\cdot} = (\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot}) + (y_{ij} - \bar{y}_{i\cdot})$
and expand the square:

$$
\begin{aligned}
\mathit{SS}_T
&= \sum_{i=1}^a\sum_{j=1}^n \bigl[(\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot}) + (y_{ij} - \bar{y}_{i\cdot})\bigr]^2 \\
&= \underbrace{\sum_{i=1}^a\sum_{j=1}^n (\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot})^2}_{\mathit{SS}_{\mathrm{Treatments}}}
 + 2\sum_{i=1}^a\sum_{j=1}^n (\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot})(y_{ij} - \bar{y}_{i\cdot})
 + \underbrace{\sum_{i=1}^a\sum_{j=1}^n (y_{ij} - \bar{y}_{i\cdot})^2}_{\mathit{SS}_E} \\
&= \mathit{SS}_{\mathrm{Treatments}} + \mathit{SS}_E
 + 2\sum_{i=1}^a (\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot})\left[\sum_{j=1}^n (y_{ij} - \bar{y}_{i\cdot})\right] \\
&= \mathit{SS}_{\mathrm{Treatments}} + \mathit{SS}_E ,
\end{aligned}
$$

because $\sum_{j=1}^n (y_{ij} - \bar{y}_{i\cdot}) = 0$ for every $i$.
:::

The degrees of freedom are $N-1$, $a-1$, and $N-a$, which also add up. The
mean squares are $\mathit{MS}_{\mathrm{Treatments}} = \mathit{SS}_{\mathrm{Treatments}}/(a-1)$
and $\mathit{MS}_E = \mathit{SS}_E/(N-a)$.

## Expected mean squares

:::{prf:theorem} Expected mean squares
:label: thm-one-way-ems

Under {prf:ref}`asm-one-way`,

$$
\begin{gathered}
\E[\mathit{MS}_E] = \sigma^2, \\
\E[\mathit{MS}_{\mathrm{Treatments}}] = \sigma^2 + \frac{n}{a-1}\sum_{i=1}^a\tau_i^2 .
\end{gathered}
$$
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

**Error mean square.** Expanding the square and using
$\sum_j y_{ij}\bar{y}_{i\cdot} = n\bar{y}_{i\cdot}^2 = y_{i\cdot}^2/n$,

$$
\begin{aligned}
\mathit{SS}_E
&= \sum_{i=1}^a\sum_{j=1}^n \bigl(y_{ij}^2 - 2y_{ij}\bar{y}_{i\cdot} + \bar{y}_{i\cdot}^2\bigr) \\
&= \sum_{i=1}^a\sum_{j=1}^n y_{ij}^2 - 2n\sum_{i=1}^a\bar{y}_{i\cdot}^2 + n\sum_{i=1}^a\bar{y}_{i\cdot}^2 \\
&= \sum_{i=1}^a\sum_{j=1}^n y_{ij}^2 - \frac1n\sum_{i=1}^a y_{i\cdot}^2 .
\end{aligned}
$$

Substituting $y_{ij} = \mu + \tau_i + \varepsilon_{ij}$ and using
$\E[\varepsilon_{ij}] = 0$, $\E[\varepsilon_{ij}^2] = \sigma^2$, and the
independence of the errors,

$$
\begin{aligned}
\E\Bigl[\sum_{i=1}^a\sum_{j=1}^n y_{ij}^2\Bigr] &= n\sum_{i=1}^a(\mu+\tau_i)^2 + N\sigma^2, \\
\E\Bigl[\frac1n\sum_{i=1}^a y_{i\cdot}^2\Bigr]
&= \frac1n\sum_{i=1}^a \E\Bigl[\Bigl(n(\mu+\tau_i) + \sum_{j=1}^n\varepsilon_{ij}\Bigr)^2\Bigr] \\
&= n\sum_{i=1}^a(\mu+\tau_i)^2 + a\sigma^2 .
\end{aligned}
$$

Hence $\E[\mathit{SS}_E] = (N-a)\sigma^2$ and $\E[\mathit{MS}_E] = \sigma^2$.

**Treatment mean square.** Since
$\sum_i\bar{y}_{i\cdot}\bar{y}_{\cdot\cdot} = a\bar{y}_{\cdot\cdot}^2$,

$$
\begin{aligned}
\mathit{MS}_{\mathrm{Treatments}}
&= \frac{n}{a-1}\sum_{i=1}^a \bigl(\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot}\bigr)^2 \\
&= \frac{n}{a-1}\left[\sum_{i=1}^a\bar{y}_{i\cdot}^2 - a\,\bar{y}_{\cdot\cdot}^2\right].
\end{aligned}
$$

Each treatment mean satisfies
$\bar{y}_{i\cdot} \sim \mathcal{N}(\mu+\tau_i,\ \sigma^2/n)$, and, because
$\sum_i\tau_i = 0$, the grand mean satisfies
$\bar{y}_{\cdot\cdot} \sim \mathcal{N}(\mu,\ \sigma^2/N)$. Using
$\E[W^2] = (\E[W])^2 + \Var(W)$,

$$
\begin{aligned}
\E[\mathit{MS}_{\mathrm{Treatments}}]
&= \frac{n}{a-1}\left\{\sum_{i=1}^a\left[(\mu+\tau_i)^2 + \frac{\sigma^2}{n}\right] - a\left[\mu^2 + \frac{\sigma^2}{an}\right]\right\} \\
&= \sigma^2 + \frac{n}{a-1}\left[\sum_{i=1}^a(\mu+\tau_i)^2 - a\mu^2\right] \\
&= \sigma^2 + \frac{n}{a-1}\left(\sum_{i=1}^a\tau_i^2 + 2\mu\sum_{i=1}^a\tau_i\right) \\
&= \sigma^2 + \frac{n}{a-1}\sum_{i=1}^a\tau_i^2 .
\end{aligned}
$$
:::

$\mathit{MS}_E$ estimates $\sigma^2$ whether or not $H_0$ holds, whereas
$\mathit{MS}_{\mathrm{Treatments}}$ estimates $\sigma^2$ only under $H_0$ and
is inflated by treatment differences otherwise. Their ratio is therefore a
natural test statistic.

## Distributions of the sums of squares

:::{prf:theorem} Distributions in one-way ANOVA
:label: thm-one-way-distributions

Under {prf:ref}`asm-one-way`:

1. $\mathit{SS}_E/\sigma^2 \sim \chi^2(N-a)$;
2. $\mathit{SS}_{\mathrm{Treatments}}$ and $\mathit{SS}_E$ are independent;
   and
3. if $H_0$ is true, $\mathit{SS}_{\mathrm{Treatments}}/\sigma^2 \sim \chi^2(a-1)$.
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

**Part 1.** Within treatment $i$, the observations $y_{i1},\ldots,y_{in}$ are
i.i.d. $\mathcal{N}(\mu+\tau_i, \sigma^2)$. By {prf:ref}`ex-sample-variance`,

$$
\frac{1}{\sigma^2}\sum_{j=1}^n (y_{ij} - \bar{y}_{i\cdot})^2 \sim \chi^2(n-1),
$$

and this within-group sum of squares is independent of $\bar{y}_{i\cdot}$.
The $a$ groups are independent, so by the additivity of independent $\chi^2$
variables,

$$
\begin{aligned}
\frac{\mathit{SS}_E}{\sigma^2} &= \sum_{i=1}^a \frac{1}{\sigma^2}\sum_{j=1}^n (y_{ij} - \bar{y}_{i\cdot})^2 \\
&\sim \chi^2\bigl(a(n-1)\bigr) = \chi^2(N-a).
\end{aligned}
$$

**Part 2.** $\mathit{SS}_{\mathrm{Treatments}}$ is a function of the treatment
means $\bar{y}_{1\cdot},\ldots,\bar{y}_{a\cdot}$, and $\mathit{SS}_E$ is a
function of the within-group sums of squares. By part 1, each within-group
sum of squares is independent of its own group mean, and the groups are
independent of each other, so the two collections, and hence
$\mathit{SS}_{\mathrm{Treatments}}$ and $\mathit{SS}_E$, are independent.

**Part 3.** Under $H_0$, $y_{ij} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(\mu,\sigma^2)$,
so the treatment means are i.i.d. $\mathcal{N}(\mu, \sigma^2/n)$. Let

$$
z_i = \frac{\bar{y}_{i\cdot} - \mu}{\sqrt{\sigma^2/n}} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,1),
\qquad i = 1,\ldots,a,
$$

and $\bar{z} = \frac1a\sum_i z_i$. Since $\bar{y}_{\cdot\cdot}$ is the
average of the treatment means,
$z_i - \bar{z} = (\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot})/\sqrt{\sigma^2/n}$, and

$$
\begin{aligned}
\sum_{i=1}^a z_i^2
&= \sum_{i=1}^a (z_i - \bar{z})^2 + a\,\bar{z}^2 \\
&= \frac{\mathit{SS}_{\mathrm{Treatments}}}{\sigma^2} + \frac{1}{a}\Bigl(\sum_{i=1}^a z_i\Bigr)^2 \\
&= \bz\T\Bigl(\bI_a - \tfrac1a\mathbf{J}_a\Bigr)\bz + \bz\T\Bigl(\tfrac1a\mathbf{J}_a\Bigr)\bz ,
\end{aligned}
$$

where $\bz = (z_1,\ldots,z_a)\T$ and $\mathbf{J}_a$ is the $a\times a$ matrix
of ones. The two matrices have ranks $a-1$ and $1$, which add up to $a$, so
by Cochran's theorem ({prf:ref}`thm-cochran`)

$$
\frac{\mathit{SS}_{\mathrm{Treatments}}}{\sigma^2} = \sum_{i=1}^a (z_i - \bar{z})^2 \sim \chi^2(a-1).
$$
:::

## The $F$-test

By {prf:ref}`thm-one-way-distributions` and {prf:ref}`def-f`, under $H_0$

$$
\begin{aligned}
F_0 &= \frac{\mathit{MS}_{\mathrm{Treatments}}}{\mathit{MS}_E} \\
    &= \frac{\bigl(\mathit{SS}_{\mathrm{Treatments}}/\sigma^2\bigr)/(a-1)}{\bigl(\mathit{SS}_E/\sigma^2\bigr)/(N-a)}
    \ \sim\ F(a-1,\ N-a).
\end{aligned}
$$ (eq-one-way-f)

$H_0$ is rejected at level $\alpha$ when $F_0 > F_{1-\alpha;\,a-1,\,N-a}$.
The results are summarized in the ANOVA table.

| Source of variation | Sum of squares | Degrees of freedom | Mean square | $F_0$ |
|---|---|---|---|---|
| Treatments | $\mathit{SS}_{\mathrm{Treatments}}$ | $a-1$ | $\mathit{MS}_{\mathrm{Treatments}}$ | $\mathit{MS}_{\mathrm{Treatments}}/\mathit{MS}_E$ |
| Error | $\mathit{SS}_E$ | $N-a$ | $\mathit{MS}_E$ | |
| Total | $\mathit{SS}_T$ | $N-1$ | | |

## Estimating the model parameters

The least squares estimates of the overall mean, the treatment effects, and
the treatment means are

$$
\begin{aligned}
\widehat{\mu} &= \bar{y}_{\cdot\cdot}, \\
\widehat{\tau}_i &= \bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot}, \\
\widehat{\mu}_i &= \widehat{\mu} + \widehat{\tau}_i \\
&= \bar{y}_{i\cdot},
\end{aligned}
$$

and $\sigma^2$ is estimated by $\widehat{\sigma}^2 = \mathit{MS}_E$
({prf:ref}`thm-one-way-ems`). The estimates satisfy
$\sum_i\widehat{\tau}_i = 0$, matching the constraint in
{prf:ref}`asm-one-way`.

### Confidence interval for a treatment mean

Under {prf:ref}`asm-one-way`, $\bar{y}_{i\cdot} \sim \mathcal{N}(\mu_i, \sigma^2/n)$.
It is independent of $\mathit{MS}_E$ (part 2 of
{prf:ref}`thm-one-way-distributions`), and
$(N-a)\mathit{MS}_E/\sigma^2 \sim \chi^2(N-a)$, so by {prf:ref}`def-t`

$$
T = \frac{\bar{y}_{i\cdot} - \mu_i}{\sqrt{\mathit{MS}_E/n}} \sim t(N-a).
$$

A $100(1-\alpha)\%$ confidence interval for $\mu_i$ is therefore

$$
\bar{y}_{i\cdot} \pm t_{1-\alpha/2,\,N-a}\sqrt{\frac{\mathit{MS}_E}{n}} .
$$ (eq-one-way-ci-mean)

### Confidence interval for a difference of treatment means

For $i \ne j$, $\bar{y}_{i\cdot} - \bar{y}_{j\cdot} \sim \mathcal{N}(\mu_i - \mu_j,\ 2\sigma^2/n)$,
and the same argument gives the $100(1-\alpha)\%$ confidence interval

$$
\bigl(\bar{y}_{i\cdot} - \bar{y}_{j\cdot}\bigr) \pm t_{1-\alpha/2,\,N-a}\sqrt{\frac{2\,\mathit{MS}_E}{n}} .
$$ (eq-one-way-ci-diff)

## Multiple comparisons

When the $F$-test rejects $H_0:\mu_1 = \cdots = \mu_a$, it does not say which
means differ. **Multiple comparison methods** answer that question. The
simplest is **Fisher's least significant difference (LSD)** method, which
tests every pair

$$
H_0:\ \mu_i = \mu_j \quad\text{versus}\quad H_1:\ \mu_i \ne \mu_j
$$

with the statistic

$$
t_0 = \frac{\bar{y}_{i\cdot} - \bar{y}_{j\cdot}}{\sqrt{2\,\mathit{MS}_E/n}} ,
$$

which follows $t(N-a)$ under $H_0$. Equivalently, the pair is declared
significantly different when $\lvert\bar{y}_{i\cdot} - \bar{y}_{j\cdot}\rvert$
exceeds the **least significant difference**

$$
\mathrm{LSD} = t_{1-\alpha/2,\,N-a}\sqrt{\frac{2\,\mathit{MS}_E}{n}} .
$$ (eq-lsd)

This is the same as checking whether the interval {eq}`eq-one-way-ci-diff`
excludes zero. Because many pairs are tested, each at level $\alpha$, the
probability of at least one false rejection exceeds $\alpha$; procedures
such as Tukey's method control this overall error rate.

## Unbalanced designs

When treatment $i$ has $n_i$ observations and $N = \sum_i n_i$, the
decomposition {prf:ref}`thm-one-way-decomposition` still holds with $n$
replaced by $n_i$ inside the sums. Convenient computing formulas are

$$
\begin{gathered}
\mathit{SS}_T = \sum_{i=1}^a\sum_{j=1}^{n_i} y_{ij}^2 - \frac{y_{\cdot\cdot}^2}{N}, \\
\mathit{SS}_{\mathrm{Treatments}} = \sum_{i=1}^a\frac{y_{i\cdot}^2}{n_i} - \frac{y_{\cdot\cdot}^2}{N}, \\
\mathit{SS}_E = \mathit{SS}_T - \mathit{SS}_{\mathrm{Treatments}},
\end{gathered}
$$

with $N-1$, $a-1$, and $N-a$ degrees of freedom; the $F$-test
{eq}`eq-one-way-f` is unchanged. In the intervals and in the LSD, $2/n$ is
replaced by $1/n_i + 1/n_j$:

$$
\mathrm{LSD}_{ij} = t_{1-\alpha/2,\,N-a}\sqrt{\mathit{MS}_E\left(\frac{1}{n_i} + \frac{1}{n_j}\right)} .
$$

A balanced design is preferable when it can be achieved: the test is then
less sensitive to unequal variances, and its power is largest for a given
$N$.

## Checking model adequacy

The conclusions above rely on independent, normally distributed errors with
constant variance ({prf:ref}`asm-one-way`). These assumptions are checked
with the **residuals**

$$
\begin{aligned}
e_{ij} &= y_{ij} - \widehat{y}_{ij}, \\
\widehat{y}_{ij} &= \widehat{\mu} + \widehat{\tau}_i \\
&= \bar{y}_{i\cdot} .
\end{aligned}
$$

If the model is adequate, the residuals are structureless: they show no
obvious pattern. Useful checks are:

- **Normality.** A normal probability plot (normal Q–Q plot) of the
  residuals should be roughly a straight line
  ({doc}`04-goodness-of-fit-tests`). Moderate departures matter little for the
  fixed-effects $F$-test.
- **Outliers.** The standardized residuals $d_{ij} = e_{ij}/\sqrt{\mathit{MS}_E}$
  are approximately $\mathcal{N}(0,1)$, so about 68% of them should lie in
  $[-1,1]$ and about 95% in $[-2,2]$; a value with $\lvert d_{ij}\rvert > 3$
  is a potential outlier and should be investigated.
- **Independence.** A plot of the residuals in the time order of data
  collection reveals correlation: runs of positive or negative residuals
  suggest that the errors are not independent.
- **Constant variance.** A plot of the residuals against the fitted values
  $\widehat{y}_{ij}$ should show no pattern; a funnel shape indicates that
  the variance changes with the mean.

What to do when an assumption fails (transformations, robust or
nonparametric tests) is beyond the scope of this chapter.

## Type II error, power, and sample size

When $H_0$ is false, $\mathit{SS}_{\mathrm{Treatments}}/\sigma^2$ follows a
noncentral $\chi^2$ distribution, and $F_0$ follows a noncentral $F$
distribution ({prf:ref}`def-noncentral-f`):

$$
\begin{gathered}
F_0 \sim F'\!\left(a-1,\ N-a,\ \delta\right), \\
\delta = \frac{n}{\sigma^2}\sum_{i=1}^a\tau_i^2 .
\end{gathered}
$$

When $\delta = 0$ this is the central $F$ distribution of
{eq}`eq-one-way-f`. The probability of a **type II error**, failing to
reject a false $H_0$, is

$$
\begin{aligned}
\beta &= 1 - \Prob\bigl(F_0 > F_{1-\alpha;\,a-1,\,N-a} \mid H_0 \text{ is false}\bigr) \\
&= \Prob\bigl(F_0 \le F_{1-\alpha;\,a-1,\,N-a} \mid H_0 \text{ is false}\bigr),
\end{aligned}
$$

and the **power** of the test is $1-\beta$. It increases with the number of
replicates $n$ and with the size of the treatment effects relative to
$\sigma^2$. Operating characteristic (OC) curves plot $\beta$ against the
parameter

$$
\begin{aligned}
\Phi^2 &= \frac{n\sum_{i=1}^a\tau_i^2}{a\,\sigma^2} \\
&= \frac{\delta}{a} .
\end{aligned}
$$

### Choosing the number of replicates

The **sensitivity** of an experiment is the smallest difference in means
that the experimenter wants to detect. If any two treatment means differ by
at least $D$, then $\sum_i\tau_i^2 \ge D^2/2$, so the smallest value of
$\Phi^2$ is

$$
\Phi^2 = \frac{n\,D^2}{2a\,\sigma^2} .
$$

In practice one fixes $\alpha$, the ratio $D/\sigma$, and the desired power,
and increases $n$ until the power computed from the noncentral $F$
distribution (or read from the OC curves) is large enough. Most statistical
software performs this calculation.

## Summary and Common Mistakes
### Summary

- The one-way fixed-effects model $y_{ij} = \mu + \tau_i + \varepsilon_{ij}$
  is a linear regression on $a-1$ indicator variables.
- $\mathit{SS}_T = \mathit{SS}_{\mathrm{Treatments}} + \mathit{SS}_E$ with
  $N-1 = (a-1) + (N-a)$ degrees of freedom.
- $\E[\mathit{MS}_E] = \sigma^2$ and
  $\E[\mathit{MS}_{\mathrm{Treatments}}] = \sigma^2 + n\sum_i\tau_i^2/(a-1)$.
- $F_0 = \mathit{MS}_{\mathrm{Treatments}}/\mathit{MS}_E \sim F(a-1,N-a)$
  under $H_0$ and $F'(a-1,N-a,\delta)$ otherwise, which gives the power and
  the number of replicates needed.
- $\widehat{\mu}_i = \bar{y}_{i\cdot}$ with interval
  $\bar{y}_{i\cdot} \pm t_{1-\alpha/2,N-a}\sqrt{\mathit{MS}_E/n}$; pairs of
  means are compared with the LSD.
- Residual plots check normality, outliers, independence, and constant
  variance.

### Common mistakes

- Forgetting the constraint $\sum_i\tau_i = 0$; without it the parameters
  $\mu,\tau_1,\ldots,\tau_a$ are not identifiable.
- Concluding from a significant $F$-test which treatments differ; that
  requires multiple-comparison procedures.
- Applying the balanced-design formulas to unbalanced designs without
  replacing $n$ by the group sizes $n_i$.
- Running all pairwise LSD comparisons without noting that the overall
  probability of a false rejection exceeds $\alpha$.
- Writing the power as $1-\alpha$; the power is $1-\beta$.

## References

- Montgomery, D. C. (2017). *Design and analysis of experiments* (9th ed.). Wiley.
- Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley.
- Wikipedia contributors. (n.d.). Noncentral chi-squared distribution. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution>
- Wikipedia contributors. (n.d.). Noncentral F-distribution. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Noncentral_F-distribution>

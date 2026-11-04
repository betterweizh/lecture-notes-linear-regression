# Poisson and Count Regression

## Overview

Count responses, such as the number of events per unit of time or space,
are non-negative integers whose variance typically grows with the mean.
Poisson regression is the GLM for such data, using the Poisson distribution
and the log link ({doc}`17-generalized-linear-models`). This chapter covers
the model, its interpretation, likelihood-based tests and diagnostics, and
three extensions for data that violate its assumptions: quasi-Poisson and
negative binomial models for overdispersion, and the zero-inflated Poisson
model for excess zeros. It builds on the exponential family, the log link, and deviance ({doc}`17-generalized-linear-models`). Python code for this chapter is in {doc}`27-glm-python`.

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- state the assumptions of Poisson regression and contrast them with those
  of linear regression;
- interpret coefficients as log rate ratios and use offsets for exposure;
- compare nested models with the drop-in-deviance test and test single
  coefficients with Wald tests;
- compute Pearson and deviance residuals and assess goodness of fit; and
- detect overdispersion and choose among quasi-Poisson, negative binomial,
  and zero-inflated models.
:::

## The Poisson distribution

In a **Poisson process**, events occur independently and the number of
events in an interval depends only on its length or size. The number of
events $Y$ in one such interval follows the **Poisson distribution**:

$$
\Prob(Y = y) = \frac{e^{-\lambda}\lambda^y}{y!}, \qquad y = 0,1,2,\ldots,
$$

where $\lambda > 0$ is the mean, or expected count, per unit of time or
space. A defining property is that the mean and variance are equal:

$$
\begin{aligned}
\E[Y] &= \Var(Y) \\
&= \lambda .
\end{aligned}
$$

## The Poisson regression model

### Assumptions

- **Poisson response.** The response $Y_i$ is a count per unit of time or
  space and follows a Poisson distribution with mean $\lambda_i$.
- **Independence.** The observations are independent of one another.
- **Mean equals variance.** $\E[Y_i] = \Var(Y_i) = \lambda_i$.
- **Linearity.** The logarithm (natural log) of the mean rate is a linear
  function of the predictors:

$$
\begin{aligned}
\ln(\lambda_i) &= \beta_0 + \beta_1x_{i1} + \beta_2x_{i2} + \cdots + \beta_kx_{ik} \\
&= \bx_i\T\bbeta, \qquad i = 1,\ldots,n .
\end{aligned}
$$ (eq-poisson-model)

In GLM terms, this is the Poisson distribution with its canonical log link,
$\eta_i = \ln\mu_i$ with $\mu_i = \lambda_i$.

### Comparison with linear regression

In linear regression:

- for each level of $x$, the responses are approximately normal;
- the variance of $y$ at each level of $x$, $\sigma^2$, is the same; and
- the mean responses $\mu_{y\mid x}$ lie on a line.

In Poisson regression:

- for small $\lambda$, the response distribution is noticeably right-skewed,
  with many small values and few large ones; as $\lambda$ increases, it
  looks more and more like a normal distribution;
- the responses become more variable as the mean increases, because the
  variance equals the mean; and
- the means $\lambda_{y\mid x}$ lie on a curve, not a line, although their
  logarithms lie on a line.

### Offset and exposure

Counts are often recorded over units of different size: different lengths
of time, population sizes, or areas. If $E_i$ is the **exposure** of
observation $i$, the natural quantity to model is the rate
$\lambda_i/E_i$:

$$
\ln\!\left(\frac{\lambda_i}{E_i}\right) = \bx_i\T\bbeta
\quad\Longleftrightarrow\quad
\ln(\lambda_i) = \ln(E_i) + \bx_i\T\bbeta .
$$

The term $o_i = \ln(E_i)$ is called an **offset**: it enters the linear
predictor with its coefficient fixed at one rather than estimated.
Specifying an exposure $E_i$ is therefore equivalent to specifying the
offset $\ln E_i$.

## Interpretation: rate ratios

Consider two predictor vectors that differ only in $x_1$, by one unit. From
{eq}`eq-poisson-model`,

$$
\left.\begin{gathered}
\ln(\lambda_{x_1}) = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots \\
\ln(\lambda_{x_1+1}) = \beta_0 + \beta_1(x_1+1) + \beta_2x_2 + \cdots
\end{gathered}\right\}
\quad\Longrightarrow\quad
\frac{\lambda_{x_1+1}}{\lambda_{x_1}} = e^{\beta_1}.
$$

A one-unit increase in $x_1$, with the other predictors fixed, multiplies
the mean count by $e^{\beta_1}$, the **rate ratio** (also called relative
risk); equivalently, the mean changes by $100\,(e^{\beta_1}-1)$ percent.

## Estimation

For $n$ independent observations, the likelihood is

$$
\begin{aligned}
\mathcal{L}(\bbeta) &= \prod_{i=1}^n \Prob(Y_i = y_i) \\
&= \prod_{i=1}^n \frac{e^{-\lambda_i}\lambda_i^{y_i}}{y_i!},
\end{aligned}
$$

where each $\lambda_i = \exp(\bx_i\T\bbeta)$ depends on the predictors of
observation $i$. The log-likelihood is

$$
\begin{aligned}
\ell(\bbeta) &= \ln\mathcal{L}(\bbeta) \\
&= \sum_{i=1}^n \bigl[-\lambda_i + y_i\ln\lambda_i - \ln(y_i!)\bigr] \\
&= \sum_{i=1}^n \bigl[-\exp(\bx_i\T\bbeta) + y_i\,\bx_i\T\bbeta - \ln(y_i!)\bigr].
\end{aligned}
$$ (eq-poisson-loglik)

The term $\ln(y_i!)$ does not depend on $\bbeta$. Because the log link is
canonical, $\ell$ is strictly concave, and the maximum likelihood estimate
is found with the algorithms of {doc}`17-generalized-linear-models`.

## Hypothesis tests

### Wald test for a single coefficient

To test

$$
\begin{gathered}
H_0 :\ \beta_j = 0 \quad\text{versus} \\
H_1 :\ \beta_j \ne 0,
\end{gathered}
$$

use

$$
Z = \frac{\widehat{\beta}_j - 0}{\mathrm{SE}(\widehat{\beta}_j)} \ \overset{\text{approx.}}{\sim}\ \mathcal{N}(0,1) \quad\text{under } H_0 .
$$

A large $|Z|$ is evidence that the coefficient is not zero.

### Drop-in-deviance test

The drop-in-deviance test compares two **nested** models, i.e. all terms of
the smaller model also appear in the larger one:

$$
\begin{gathered}
H_0 :\ \text{the reduced model is adequate (the extra coefficients of the full model are zero)}, \\
H_1 :\ \text{the larger (full) model is needed}.
\end{gathered}
$$

Fit both models and compute

$$
\begin{aligned}
\text{drop in deviance} &= D_{\text{reduced}} - D_{\text{full}} \\
&= -2\bigl[\ell(\widehat{\bbeta}_{\text{reduced}}) - \ell(\widehat{\bbeta}_{\text{full}})\bigr],
\end{aligned}
$$

where $D$ is the residual deviance {eq}`eq-deviance`. Because the
saturated log-likelihood cancels, the drop in deviance equals the likelihood
ratio statistic. Under $H_0$,

$$
D_{\text{reduced}} - D_{\text{full}} \ \overset{\text{approx.}}{\sim}\ \chi^2(d),
$$

where $d$ is the difference in the number of coefficients between the two
models. A large drop in deviance favors the larger model.

## Goodness of fit

The **residual deviance** measures how far the fitted values are from the
observed counts. If the model is adequate and the counts are not too small,

$$
D \ \overset{\text{approx.}}{\sim}\ \chi^2(n-p),
$$

where $n-p$ is the residual degrees of freedom and $p$ the number of
coefficients, including the intercept. A deviance much larger than $n-p$
indicates lack of fit.

## Residuals

**Pearson residuals** divide the raw residual by the estimated standard
deviation of the response:

$$
r_{P,i} = \frac{y_i - \widehat{\lambda}_i}{\sqrt{\widehat{\lambda}_i}},
$$

where $\widehat{\lambda}_i$ is the fitted mean of observation $i$. If the
model is adequate, most Pearson residuals lie between $-2$ and $2$.

**Deviance residuals** are the signed square roots of each observation's
contribution to the deviance:

$$
r_{D,i} = \sign(y_i-\widehat{\lambda}_i)\sqrt{2\left[y_i\ln\!\left(\frac{y_i}{\widehat{\lambda}_i}\right) - (y_i-\widehat{\lambda}_i)\right]},
$$

with the convention $y_i\ln(y_i/\widehat{\lambda}_i) = 0$ when $y_i = 0$, and
where

$$
\sign(x) = \begin{cases} 1, & x > 0, \\ 0, & x = 0, \\ -1, & x < 0. \end{cases}
$$

The residual deviance is the sum of squared deviance residuals,
$D = \sum_i r_{D,i}^2$.

## Overdispersion

**Overdispersion** means that the response varies more than the model
implies. Under a Poisson model, the mean and variance of the response should
be about the same within groups of similar observations; in practice, the
variance often exceeds the mean. Ignoring overdispersion leads to standard
errors that are too small and tests that are too liberal.

Two common remedies are:

- use an estimated dispersion factor to inflate the standard errors
  (quasi-Poisson); or
- use a negative binomial regression model.

### Quasi-Poisson

Estimate the **dispersion parameter** $\phi$, sometimes called the Pearson
dispersion (Hilbe, 2011), by

$$
\widehat{\phi} = \frac{\sum_{i=1}^n r_{P,i}^2}{n-p},
$$

where $n$ is the number of observations, $p$ the number of coefficients,
and $n-p$ the residual degrees of freedom. If the model is adequate,
$\sum_i r_{P,i}^2$ is approximately $\chi^2(n-p)$, so $\widehat{\phi}$ should
be close to one; it is noticeably larger than one when overdispersion is
present.

The quasi-Poisson approach keeps the Poisson estimates but multiplies the
variances by $\widehat{\phi}$, so the standard errors are larger than the
likelihood approach implies:

$$
\mathrm{SE}_Q(\widehat{\beta}_j) = \sqrt{\widehat{\phi}}\ \mathrm{SE}(\widehat{\beta}_j).
$$

The subscript $Q$ stands for "quasi": inflating the variances by
$\widehat{\phi}$ is an ad hoc adjustment that does not correspond to a full
likelihood ("quasi-likelihood").

### Poisson versus quasi-Poisson inference

| | Poisson | Quasi-Poisson |
|---|---|---|
| Estimate | $\widehat{\beta}$ | $\widehat{\beta}$ |
| Standard error | $\mathrm{SE}(\widehat{\beta})$ | $\mathrm{SE}_Q(\widehat{\beta}) = \sqrt{\widehat{\phi}}\,\mathrm{SE}(\widehat{\beta})$ |
| Wald-type test statistic | $Z = \widehat{\beta}/\mathrm{SE}(\widehat{\beta})$ | $t = \widehat{\beta}/\mathrm{SE}_Q(\widehat{\beta})$ |
| Confidence interval | $\widehat{\beta} \pm z_{1-\alpha/2}\,\mathrm{SE}(\widehat{\beta})$ | $\widehat{\beta} \pm t_{1-\alpha/2,\,n-p}\,\mathrm{SE}_Q(\widehat{\beta})$ |
| Drop-in-deviance test | $\chi^2 = D_{\text{reduced}} - D_{\text{full}}$ | $F = \dfrac{(D_{\text{reduced}} - D_{\text{full}})/d}{\widehat{\phi}}$ |

In the quasi-Poisson column, $t$ approximately follows a $t$ distribution
with $n-p$ degrees of freedom, and $F$ approximately follows an $F$
distribution with $d$ (the difference in the number of coefficients) and
$n-p$ (the residual degrees of freedom of the full model) degrees of
freedom.

### Negative binomial regression

The negative binomial model has several advantages:

- it introduces a second parameter in addition to the mean, which gives the
  model more flexibility;
- unlike the quasi-Poisson approach, it specifies an explicit likelihood;
  and
- negative binomial random variables take non-negative integer values,
  consistent with count data.

#### The negative binomial distribution

If $Y$ is the number of failures before the $r$-th success in independent
Bernoulli trials with success probability $p_s$, then $Y$ follows a
**negative binomial distribution**:

$$
\Prob(Y = y) = \binom{y + r - 1}{r - 1}(1-p_s)^y\,p_s^{\,r},
\qquad y = 0,1,2,\ldots,
$$

with $0 < p_s \le 1$. Its mean and variance are

$$
\begin{aligned}
\E[Y] &= \mu \\
&= \frac{r(1-p_s)}{p_s}, \\
\Var(Y) &= \frac{r(1-p_s)}{p_s^2} \\
&= \mu + \frac{\mu^2}{r}.
\end{aligned}
$$

**Relationship with the Poisson distribution.** If
$Y\mid\lambda \sim \mathrm{Poisson}(\lambda)$ and $\lambda$ follows a gamma
distribution with shape $r$ and scale $(1-p_s)/p_s$, then
$Y \sim \mathrm{NegBin}(r, p_s)$. The negative binomial is therefore a
Poisson distribution whose mean varies randomly across observations. The
extra variance $\mu^2/r$ measures the overdispersion; it approaches zero as
$r\to\infty$, so smaller values of $r$ indicate greater overdispersion.

#### The regression model

Negative binomial regression keeps the log link of
{eq}`eq-poisson-model`, $\ln\mu_i = \bx_i\T\bbeta$, and replaces the Poisson
variance by $\Var(Y_i) = \mu_i + \mu_i^2/r$. The coefficients and $r$ (or
the dispersion $\alpha = 1/r$) are estimated by maximum likelihood, and the
coefficients keep their rate-ratio interpretation.

## Zero-inflated Poisson model

Some count data contain more zeros than a Poisson distribution allows,
because two processes generate them: one that always produces a zero
("structural" zeros) and a Poisson process that can produce zeros by
chance.

### The zero-inflated Poisson distribution

With probability $\pi$ an observation is a structural zero; with
probability $1-\pi$ it is drawn from $\mathrm{Poisson}(\lambda)$:

$$
\Prob(Y = y) =
\begin{cases}
\pi + (1-\pi)\,e^{-\lambda}, & y = 0, \\[4pt]
(1-\pi)\,\dfrac{\lambda^y e^{-\lambda}}{y!}, & y = 1,2,3,\ldots
\end{cases}
$$

The mean and variance of a zero-inflated Poisson (ZIP) variable are

$$
\begin{gathered}
\E[Y] = (1-\pi)\,\lambda, \\
\Var(Y) = \lambda(1-\pi)(1+\pi\lambda).
\end{gathered}
$$

### Zero-inflated Poisson regression

The ZIP regression model is a mixture (latent class) model with two linear
predictors:

$$
\begin{gathered}
\text{count part:} \qquad  \ln(\lambda_i) = \beta_0 + \beta_1x_{i1} + \beta_2x_{i2} + \cdots, \\
\text{zero part:} \qquad  \logit(\pi_i) = \gamma_0 + \gamma_1z_{i1} + \gamma_2z_{i2} + \cdots,
\end{gathered}
$$

where $\pi_i$ is the probability that observation $i$ belongs to the
structural-zero group, and the predictors $z_{i1}, z_{i2},\ldots$ of the
zero part may differ from those of the count part. The zero part describes
membership of the "always zero" group versus the "count" group.

### Vuong test

Ordinary likelihood ratio tests require nested models, but the Poisson and
ZIP models are not nested in the usual sense. The **Vuong test** compares the
predicted probabilities of two non-nested models and is commonly used to
choose between them.

## Summary and Common Mistakes
### Summary

- Poisson regression models $\ln\E[Y_i] = \bx_i\T\bbeta$, assuming the mean
  equals the variance.
- $e^{\beta_j}$ is a rate ratio; offsets $\ln E_i$ account for exposure.
- Nested models are compared with the drop-in-deviance test; single
  coefficients with Wald tests.
- Overdispersion is handled with quasi-Poisson standard errors or negative
  binomial regression; excess zeros with zero-inflated models.

### Common mistakes

- Reading $e^{\beta_1}$ as an additive change or as "$e^{\beta_1}$ percent";
  it is a multiplicative change in the mean.
- Including $\ln(E_i)$ as an ordinary predictor with an estimated
  coefficient instead of as an offset.
- Treating the residual deviance as $-2\ell(\widehat{\bbeta})$; only
  differences of deviances equal differences of $-2\ell$.
- Ignoring overdispersion, which makes standard errors too small.

## References

- Hilbe, J. M. (2011). *Negative binomial regression* (2nd ed.). Cambridge University Press. <https://doi.org/10.1017/CBO9780511973420>
- Roback, P., & Legler, J. (2021). *Beyond multiple linear regression: Applied generalized linear models and multilevel models in R*. CRC Press. <https://bookdown.org/roback/bookdown-BeyondMLR/>
- StataCorp. (n.d.). *Stata documentation* [Software documentation]. <https://www.stata.com/features/documentation/>

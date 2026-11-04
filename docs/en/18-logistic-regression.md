# Logistic Regression

## Overview

Many responses are binary: a traveler chooses transit or not, a patient
recovers or not, a loan defaults or not. **Logistic regression** models the
probability of the outcome $Y = 1$ as a function of predictors. It is the GLM
with a Bernoulli (or binomial) response and the canonical logit link
({doc}`17-generalized-linear-models`), and it is also the simplest
**discrete choice model**: with two alternatives, the multinomial logit model
of {doc}`21-multinomial-choice-models` reduces to it. This chapter defines the
model, interprets its coefficients as odds ratios, derives the maximum
likelihood equations and the IRLS algorithm, and describes inference and
model assessment. It builds on the exponential family, canonical link, deviance, and IRLS ({doc}`17-generalized-linear-models`) and on the large-sample properties of the MLE ({doc}`02-sampling-distributions-and-limit-theorems`).

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- explain why a linear model for a probability is inadequate and write the
  logistic regression model;
- interpret coefficients as log odds ratios and compute marginal effects;
- derive the log-likelihood, score, and Hessian, and describe the
  Newton–Raphson (IRLS) algorithm;
- recognize separation, which prevents the maximum likelihood estimate from
  existing;
- carry out Wald and likelihood ratio tests and assess fit; and
- relate logistic regression to the latent-variable and random-utility
  formulations.
:::

## Notation

| Symbol | Meaning |
|---|---|
| $Y_i \in \{0,1\}$ | binary response of observation $i$ |
| $\pi_i = \Prob(Y_i = 1\mid\bx_i)$ | success probability |
| $\eta_i = \bx_i\T\bbeta$ | linear predictor ($\bx_i$ includes a leading 1) |
| $\logit(\pi) = \ln\{\pi/(1-\pi)\}$ | log odds |
| $\mathbf{W} = \diag\bigl(\pi_i(1-\pi_i)\bigr)$ | IRLS weight matrix |

## The model

### Why not a linear model?

A linear model $\pi_i = \bx_i\T\bbeta$ for a probability (the *linear
probability model*) can produce fitted values outside $[0,1]$, and the
variance $\Var(Y_i) = \pi_i(1-\pi_i)$ depends on the mean, violating the
homoskedasticity assumption of {doc}`07-linear-regression-model`. Logistic
regression maps the linear predictor into $(0,1)$ instead.

### Logistic regression

:::{prf:definition} Logistic regression model
:label: def-logistic

The responses $Y_1,\ldots,Y_n$ are independent with
$Y_i\mid\bx_i \sim \mathrm{Bernoulli}(\pi_i)$ and

$$
\begin{aligned}
\logit(\pi_i) &= \ln\frac{\pi_i}{1-\pi_i} \\
&= \bx_i\T\bbeta,
\end{aligned}
$$

equivalently

$$
\begin{aligned}
\pi_i &= \frac{\exp(\bx_i\T\bbeta)}{1 + \exp(\bx_i\T\bbeta)} \\
&= \frac{1}{1 + \exp(-\bx_i\T\bbeta)} .
\end{aligned}
$$ (eq-logistic)
:::

The function $\pi(\eta) = 1/(1+e^{-\eta})$ is the logistic CDF
({doc}`01-probability-and-random-vectors`); it is S-shaped, increasing, and
symmetric about $\eta = 0$, where $\pi = 0.5$. By
{prf:ref}`ex-bernoulli-canonical`, the logit is the canonical link of the
Bernoulli distribution, so the model is a GLM with canonical link and
variance function $V(\pi) = \pi(1-\pi)$.

### Latent-variable and random-utility formulations

Suppose a continuous latent variable $Y^{*} = \bx\T\bbeta + \varepsilon$ is
observed only through $Y = \mathbb{I}(Y^{*} > 0)$. If $\varepsilon$ follows
the standard logistic distribution, then

$$
\begin{aligned}
\Prob(Y = 1\mid\bx)
&= \Prob(\varepsilon > -\bx\T\bbeta) \\
&= 1 - F(-\bx\T\bbeta) \\
&= F(\bx\T\bbeta),
\end{aligned}
$$

which is {eq}`eq-logistic`. If instead $\varepsilon \sim \mathcal{N}(0,1)$,
the result is the **probit model** $\Prob(Y=1\mid\bx) = \Phi(\bx\T\bbeta)$.
In choice modeling, $Y^{*}$ is the utility difference between two
alternatives; with independent Gumbel errors for each utility, their
difference is logistic, and the binary logit model is the two-alternative
case of the multinomial logit model ({prf:ref}`thm-mnl`).

## Interpreting the coefficients

### Odds and odds ratios

The **odds** of success are $\pi/(1-\pi)$. Under the model, the log odds are
linear in the predictors, so increasing $x_j$ by one unit, with the other
predictors held fixed, changes the log odds by $\beta_j$:

$$
\begin{aligned}
\frac{\text{odds}(x_j + 1)}{\text{odds}(x_j)}
&= \frac{\exp(\beta_0 + \cdots + \beta_j(x_j+1) + \cdots)}{\exp(\beta_0 + \cdots + \beta_jx_j + \cdots)} \\
&= e^{\beta_j} .
\end{aligned}
$$

The **odds ratio** $e^{\beta_j}$ is constant in $x_j$: a one-unit increase
multiplies the odds by $e^{\beta_j}$. It does not multiply the probability by
$e^{\beta_j}$; the change in probability depends on where on the S-curve the
observation lies.

### Marginal effects

For a continuous predictor, the derivative of the probability is

$$
\frac{\partial\pi}{\partial x_j} = \beta_j\,\pi(1-\pi),
$$

which is largest at $\pi = 0.5$, where it equals $\beta_j/4$. Average
marginal effects average this quantity over the observations.

## Estimation

### Likelihood, score, and Hessian

For independent observations, the likelihood is
$\prod_i\pi_i^{y_i}(1-\pi_i)^{1-y_i}$, and the log-likelihood, with
$\eta_i = \bx_i\T\bbeta$, is

$$
\begin{aligned}
\ell(\bbeta)
&= \sum_{i=1}^n\bigl[y_i\ln\pi_i + (1-y_i)\ln(1-\pi_i)\bigr] \\
&= \sum_{i=1}^n\bigl[y_i\,\eta_i - \ln\bigl(1 + e^{\eta_i}\bigr)\bigr].
\end{aligned}
$$ (eq-logistic-loglik)

Using $\partial\ln(1+e^{\eta})/\partial\eta = \pi$ and
$\partial\pi/\partial\eta = \pi(1-\pi)$,

$$
\begin{aligned}
\nabla\ell(\bbeta) &= \sum_{i=1}^n (y_i - \pi_i)\,\bx_i \\
&= \bX\T(\by - \boldsymbol\pi), \\
\frac{\partial^2\ell}{\partial\bbeta\,\partial\bbeta\T} &= -\sum_{i=1}^n \pi_i(1-\pi_i)\,\bx_i\bx_i\T \\
&= -\bX\T\mathbf{W}\bX ,
\end{aligned}
$$

where $\boldsymbol\pi = (\pi_1,\ldots,\pi_n)\T$. The Hessian is negative
definite when $\bX$ has full column rank, so the log-likelihood is strictly
concave ({prf:ref}`thm-optimality`) and the maximum likelihood estimate,
when it exists, is unique. The likelihood equations
$\bX\T(\by - \widehat{\boldsymbol\pi}) = \bzero$ have no closed-form solution.

### Newton–Raphson and IRLS

Because the logit link is canonical, the Hessian does not depend on $\by$,
and Newton–Raphson coincides with Fisher scoring:

$$
\begin{aligned}
\bbeta^{(t+1)}
&= \bbeta^{(t)} + (\bX\T\mathbf{W}\bX)^{-1}\bX\T(\by - \boldsymbol\pi) \\
&= (\bX\T\mathbf{W}\bX)^{-1}\bX\T\mathbf{W}\mathbf{z},
\qquad
z_i = \eta_i + \frac{y_i - \pi_i}{\pi_i(1-\pi_i)},
\end{aligned}
$$

where $\mathbf{W}$, $\boldsymbol\pi$, and $\boldsymbol\eta$ are evaluated at
$\bbeta^{(t)}$. Each step is a weighted least squares regression of the
working response $\mathbf{z}$ on $\bX$: the IRLS algorithm of
{doc}`17-generalized-linear-models` with $g'(\pi) = 1/\{\pi(1-\pi)\}$ and
$V(\pi) = \pi(1-\pi)$. At convergence,
$\widehat{\Cov}(\widehat{\bbeta}) = (\bX\T\widehat{\mathbf{W}}\bX)^{-1}$, the
inverse Fisher information.

### Separation

If some linear combination of the predictors separates the observations
perfectly, i.e. there is $\mathbf{b}$ with $\bx_i\T\mathbf{b} > 0$ whenever
$y_i = 1$ and $\bx_i\T\mathbf{b} < 0$ whenever $y_i = 0$, then the
log-likelihood increases toward $0$ as $\bbeta = c\,\mathbf{b}$ with
$c\to\infty$, and **the maximum likelihood estimate does not exist**.
Software then reports huge coefficients and standard errors, or a
convergence warning. Remedies include removing or combining the separating
predictor, collecting more data, or using a penalized (ridge or Firth)
logistic regression.

### Grouped binomial data

When there are $m_i$ trials with $y_i$ successes at each distinct
predictor pattern, $Y_i \sim \mathrm{Binomial}(m_i, \pi_i)$ and the
log-likelihood becomes
$\sum_i\bigl[y_i\eta_i - m_i\ln(1+e^{\eta_i})\bigr]$ plus a constant. The
estimates are the same as those obtained by expanding the data into
individual Bernoulli observations.

## Inference

By {prf:ref}`thm-mle-asymptotic`, the following large-sample procedures
apply.

- **Wald test** of $H_0:\beta_j = 0$:
  $Z = \widehat{\beta}_j/\mathrm{SE}(\widehat{\beta}_j)$, approximately
  $\mathcal{N}(0,1)$ under $H_0$, where $\mathrm{SE}(\widehat{\beta}_j)$ is the
  square root of the $j$-th diagonal entry of $(\bX\T\widehat{\mathbf{W}}\bX)^{-1}$.
- **Confidence interval for an odds ratio:** exponentiate the Wald interval,
  $\exp\bigl(\widehat{\beta}_j \pm z_{1-\alpha/2}\,\mathrm{SE}(\widehat{\beta}_j)\bigr)$.
- **Likelihood ratio test** for nested models:
  $G^2 = -2\bigl[\ell(\widehat{\bbeta}_{\text{reduced}}) - \ell(\widehat{\bbeta}_{\text{full}})\bigr]$,
  approximately $\chi^2(d)$ under $H_0$, where $d$ is the number of
  coefficients removed. It is the drop-in-deviance test of
  {doc}`19-poisson-and-count-regression` applied to binary data, and is usually
  more reliable than the Wald test.

## Model assessment

### Deviance

The deviance of a logistic regression is

$$
D = 2\sum_{i=1}^n\left[y_i\ln\frac{y_i}{\widehat{\pi}_i} + (1-y_i)\ln\frac{1-y_i}{1-\widehat{\pi}_i}\right],
$$

with $0\ln 0 = 0$. For ungrouped binary data the saturated log-likelihood
is zero, so $D = -2\ell(\widehat{\bbeta})$; in this case $D$ does **not** have
an approximate $\chi^2(n-p)$ distribution and should not be used as an
absolute goodness-of-fit test, although differences of deviances remain
valid. For grouped data with large $m_i$, $D \approx \chi^2(\text{number of groups} - p)$.

### Other measures

- **Pseudo-$R^2$.** McFadden's
  $R^2_{\mathrm{McF}} = 1 - \ell(\widehat{\bbeta})/\ell(\widehat{\beta}_0)$
  compares the model with the intercept-only model; it is the likelihood ratio
  index $\rho^2$ of {doc}`21-multinomial-choice-models` with a different
  baseline.
- **Hosmer–Lemeshow test.** Observations are grouped by fitted probability
  (often into ten groups) and observed and expected counts are compared
  with a $\chi^2$ statistic.
- **Classification.** Predicting $\widehat{Y} = 1$ when
  $\widehat{\pi} > c$ for a threshold $c$ (often $0.5$) gives a confusion
  matrix, sensitivity, and specificity; the ROC curve and its area (AUC)
  summarize performance over all thresholds.
- **Residuals.** Pearson residuals
  $(y_i - \widehat{\pi}_i)/\sqrt{\widehat{\pi}_i(1-\widehat{\pi}_i)}$ and
  deviance residuals are defined as in
  {doc}`19-poisson-and-count-regression`.

## Summary and Common Mistakes
### Summary

- Logistic regression models $\logit(\pi_i) = \bx_i\T\bbeta$ for a Bernoulli
  response; it is the canonical-link GLM for binary data and the binary
  case of the multinomial logit model.
- $e^{\beta_j}$ is an odds ratio; $\partial\pi/\partial x_j = \beta_j\pi(1-\pi)$.
- The score is $\bX\T(\by-\boldsymbol\pi)$ and the Hessian $-\bX\T\mathbf{W}\bX$;
  the log-likelihood is strictly concave and is maximized by IRLS.
- Under separation the MLE does not exist.
- Use Wald and likelihood ratio tests; assess fit with pseudo-$R^2$,
  Hosmer–Lemeshow, and classification measures.

### Common mistakes

- Reading $e^{\beta_j}$ as a ratio of probabilities (a relative risk); it is
  a ratio of odds, which is close to a relative risk only when
  probabilities are small.
- Using the residual deviance of ungrouped binary data as a $\chi^2$
  goodness-of-fit test.
- Ignoring convergence warnings caused by separation.
- Comparing logit and probit coefficients directly; they are on different
  scales (logit coefficients are roughly $1.6$–$1.8$ times probit
  coefficients).

## References

- Agresti, A. (2013). *Categorical data analysis* (3rd ed.). Wiley.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied logistic regression* (3rd ed.). Wiley. <https://doi.org/10.1002/9781118548387>
- McCullagh, P., & Nelder, J. A. (1989). *Generalized linear models* (2nd ed.). Chapman & Hall.
- Train, K. E. (2009). *Discrete choice methods with simulation* (2nd ed.). Cambridge University Press. <https://doi.org/10.1017/CBO9780511805271>

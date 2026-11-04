# Generalized Linear Models

## Overview

The normal linear model assumes a continuous, normally distributed response
whose mean is a linear function of the predictors. Many responses are not
like this: counts are non-negative integers, binary outcomes take only the
values $0$ and $1$, and their variances depend on their means. A
**generalized linear model (GLM)** keeps the linear predictor but allows the
response to follow any distribution in the exponential family and connects
the mean to the linear predictor through a link function. This chapter
develops the framework used by the binary, count, ordinal, and multinomial models
in the rest of Part IV. It builds on maximum likelihood ({doc}`08-ols-and-ml-estimation`), the large-sample properties of the MLE ({doc}`02-sampling-distributions-and-limit-theorems`), and gradients, Hessians, and Taylor expansions ({doc}`06-matrix-calculus`).

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- identify the random component, linear predictor, and link function of a
  GLM;
- write common distributions in canonical exponential family form and derive
  their mean and variance from the cumulant function $b(\theta)$;
- define the canonical link and explain its advantages;
- write the GLM log-likelihood and define the deviance; and
- describe the Newton–Raphson, Fisher scoring, and iteratively reweighted
  least squares (IRLS) algorithms.
:::

## Notation

| Symbol | Meaning |
|---|---|
| $Y_i$, $y_i$ | random response and its observed value, $i = 1,\ldots,n$ |
| $\bx_i \in \R^p$ | predictor vector of observation $i$ (first entry $1$ for an intercept) |
| $\mu_i = \E[Y_i\mid\bx_i]$ | conditional mean |
| $\eta_i = \bx_i\T\bbeta$ | linear predictor |
| $g$ | link function, $g(\mu_i) = \eta_i$ |
| $\theta_i$ | canonical (natural) parameter |
| $\phi$ | dispersion parameter |
| $b(\theta)$, $c(y,\phi)$ | known functions defining the distribution |
| $V(\mu)$ | variance function, $\Var(Y) = \phi\,V(\mu)$ |
| $\ell(\bbeta)$ | log-likelihood |

## From linear models to GLMs

A normal linear model has two components:

- **Random component.** Given $\bx$, the response $Y$ is continuous and
  normally distributed with mean $\mu(\bx) = \E[Y\mid\bx]$.
- **Systematic component.** The mean is linear in the predictors:
  $\mu(\bx) = \bx\T\bbeta$.

A GLM generalizes both components:

- **Random component.** $Y\mid\bx$ follows a distribution from the
  exponential family.
- **Systematic component and link.** A link function $g$ relates the mean to
  the linear predictor,

  $$
  \begin{aligned}
  g\bigl(\mu(\bx)\bigr) &= \eta \\
  &= \bx\T\bbeta .
  \end{aligned}
  $$

The normal linear model is the special case with a normal response and the
identity link $g(\mu) = \mu$.

## The exponential family

### General definition

:::{prf:definition} Exponential family
:label: def-exp-family

A family of distributions $\{P_{\boldsymbol\vartheta} : \boldsymbol\vartheta \in \Theta\}$,
$\Theta \subseteq \R^m$, is an $m$-parameter **exponential family** on
$\R^q$ if there are real-valued functions
$\eta_1,\ldots,\eta_m$ and $B$ of $\boldsymbol\vartheta$, and
$T_1,\ldots,T_m$ and $h$ of $x\in\R^q$, such that the density (probability
mass function or probability density function) of $P_{\boldsymbol\vartheta}$
can be written as

$$
p_{\boldsymbol\vartheta}(x)
= \exp\Bigl[\sum_{j=1}^m \eta_j(\boldsymbol\vartheta)\,T_j(x) - B(\boldsymbol\vartheta)\Bigr]\,h(x).
$$
:::

In this definition only, $\eta_j(\cdot)$ denotes the natural parameter
functions of the family, not the linear predictor.

:::{prf:example} The normal distribution
:label: ex-normal-exp-family

Let $X\sim\mathcal{N}(\mu,\sigma^2)$ with $\boldsymbol\vartheta = (\mu,\sigma^2)$.
The density is

$$
p_{\boldsymbol\vartheta}(x)
= \exp\!\left(\frac{\mu}{\sigma^2}x - \frac{1}{2\sigma^2}x^2 - \frac{\mu^2}{2\sigma^2}\right)\frac{1}{\sigma\sqrt{2\pi}},
$$

a two-parameter exponential family with

$$
\begin{gathered}
\eta_1 = \frac{\mu}{\sigma^2}, \qquad  \eta_2 = -\frac{1}{2\sigma^2}, \\
T_1(x) = x, \qquad  T_2(x) = x^2, \\
B(\boldsymbol\vartheta) = \frac{\mu^2}{2\sigma^2} + \ln\bigl(\sigma\sqrt{2\pi}\bigr), \qquad  h(x) = 1 .
\end{gathered}
$$

When $\sigma^2$ is known, it becomes a one-parameter exponential family on
$\R$ with

$$
\begin{gathered}
\eta = \frac{\mu}{\sigma^2}, \\
T(x) = x, \\
B(\mu) = \frac{\mu^2}{2\sigma^2}, \\
h(x) = \frac{e^{-x^2/(2\sigma^2)}}{\sigma\sqrt{2\pi}} .
\end{gathered}
$$
:::

Other members of the exponential family include:

- discrete distributions: Bernoulli, binomial (with known number of trials),
  Poisson, and negative binomial (with known shape);
- continuous distributions: gamma, inverse gamma, inverse Gaussian,
  chi-square, and beta.

### Canonical one-parameter form

GLMs use the following one-parameter form for a scalar response
$y\in\R$:

$$
f_\theta(y) = \exp\left(\frac{y\theta - b(\theta)}{\phi} + c(y,\phi)\right)
$$ (eq-canonical-exp-family)

for known functions $b(\cdot)$ and $c(\cdot,\cdot)$.

- If $\phi$ is known, this is a one-parameter exponential family with
  **canonical parameter** $\theta$.
- If $\phi$ is unknown, the family may or may not be a two-parameter
  exponential family. $\phi$ is called the **dispersion parameter**.
- In this chapter we assume that $\phi$ is known.

:::{prf:example} Normal distribution with known variance
:label: ex-normal-canonical

$$
\begin{aligned}
f_\theta(y)
&= \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right) \\
&= \exp\left\{\frac{y\mu - \tfrac12\mu^2}{\sigma^2}
   - \frac12\left(\frac{y^2}{\sigma^2} + \ln(2\pi\sigma^2)\right)\right\}.
\end{aligned}
$$

Therefore $\theta = \mu$, $\phi = \sigma^2$, $b(\theta) = \theta^2/2$, and
$c(y,\phi) = -\tfrac12\bigl(y^2/\phi + \ln(2\pi\phi)\bigr)$.
:::

:::{prf:example} Poisson distribution
:label: ex-poisson-canonical

$$
\begin{aligned}
f(y\mid\mu) &= \frac{\mu^y}{y!}e^{-\mu} \\
&= \exp\bigl[y\ln\mu - \mu - \ln(y!)\bigr], \qquad y = 0,1,2,\ldots
\end{aligned}
$$

Therefore $\theta = \ln\mu$, $b(\theta) = e^\theta$, $\phi = 1$, and
$c(y,\phi) = -\ln(y!)$.
:::

:::{prf:example} Bernoulli distribution
:label: ex-bernoulli-canonical

For $y\in\{0,1\}$ with $\Prob(Y=1) = \mu$,

$$
\begin{aligned}
f(y\mid\mu) &= \mu^y(1-\mu)^{1-y} \\
&= \exp\left[y\ln\frac{\mu}{1-\mu} + \ln(1-\mu)\right].
\end{aligned}
$$

Therefore $\theta = \ln\{\mu/(1-\mu)\}$, so $\mu = e^\theta/(1+e^\theta)$ and
$b(\theta) = -\ln(1-\mu) = \ln(1+e^\theta)$, with $\phi = 1$ and
$c(y,\phi) = 0$.
:::

| | Normal | Poisson | Bernoulli |
|---|---|---|---|
| Notation | $\mathcal{N}(\mu,\sigma^2)$ | $\mathrm{Poisson}(\mu)$ | $\mathrm{Bernoulli}(\mu)$ |
| Range of $y$ | $(-\infty,\infty)$ | $\{0,1,2,\ldots\}$ | $\{0,1\}$ |
| $\phi$ | $\sigma^2$ | $1$ | $1$ |
| $b(\theta)$ | $\theta^2/2$ | $e^\theta$ | $\ln(1+e^\theta)$ |
| $c(y,\phi)$ | $-\tfrac12\bigl(y^2/\phi + \ln(2\pi\phi)\bigr)$ | $-\ln(y!)$ | $0$ |

## Mean and variance

Let $\ell(\theta) = \ln f_\theta(Y)$ be the log-likelihood of a single
observation. Differentiating the identity $\int f_\theta(y)\,dy \equiv 1$
(a sum for discrete $Y$) once and twice with respect to $\theta$ gives two
identities:

$$
\begin{gathered}
\E\!\left[\frac{\partial\ell}{\partial\theta}\right] = 0, \\
\E\!\left[\frac{\partial^2\ell}{\partial\theta^2}\right] + \E\!\left[\left(\frac{\partial\ell}{\partial\theta}\right)^2\right] = 0 .
\end{gathered}
$$ (eq-score-identities)

### Expected value

From {eq}`eq-canonical-exp-family`,
$\ell(\theta) = \{Y\theta - b(\theta)\}/\phi + c(Y,\phi)$, so

$$
\frac{\partial\ell}{\partial\theta} = \frac{Y - b'(\theta)}{\phi}.
$$

The first identity gives $0 = \{\E[Y] - b'(\theta)\}/\phi$, hence

$$
\begin{aligned}
\E[Y] &= \mu \\
&= b'(\theta).
\end{aligned}
$$ (eq-glm-mean)

### Variance

The second derivative is $\partial^2\ell/\partial\theta^2 = -b''(\theta)/\phi$,
and by {eq}`eq-glm-mean`,
$\partial\ell/\partial\theta = (Y-\E[Y])/\phi$. The second identity becomes

$$
0 = -\frac{b''(\theta)}{\phi} + \frac{\Var(Y)}{\phi^2},
$$

hence

$$
\Var(Y) = \phi\,b''(\theta).
$$ (eq-glm-variance)

Because $\mu = b'(\theta)$, the variance can be written as a function of the
mean, $\Var(Y) = \phi\,V(\mu)$, where $V(\mu) = b''\bigl((b')^{-1}(\mu)\bigr)$
is the **variance function**. For example, $V(\mu) = 1$ for the normal,
$V(\mu) = \mu$ for the Poisson, and $V(\mu) = \mu(1-\mu)$ for the Bernoulli
distribution.

## Link functions

### Definition

The parameter of interest is $\bbeta$, and it must enter the likelihood to
be estimated by maximum likelihood. A **link function** $g$ relates the
linear predictor $\eta = \bx\T\bbeta$ to the mean $\mu$:

$$
\begin{gathered}
g(\mu) = \bx\T\bbeta, \\
\mu = g^{-1}\bigl(\bx\T\bbeta\bigr).
\end{gathered}
$$

The link is required to be monotone and differentiable, so that it can be
inverted.

Common link functions:

| Link | $g(\mu)$ | Typical use |
|---|---|---|
| identity | $\mu$ | normal responses |
| log | $\ln\mu$ | counts, positive responses |
| logit | $\ln\{\mu/(1-\mu)\}$ | binary responses |
| probit | $\Phi^{-1}(\mu)$, $\Phi$ the standard normal CDF | binary responses |
| complementary log–log | $\ln\{-\ln(1-\mu)\}$ | binary responses |
| inverse | $1/\mu$ | gamma responses |

### Canonical link

:::{prf:definition} Canonical link
:label: def-canonical-link

The link function that maps the mean to the canonical parameter,
$g(\mu) = \theta$, is called the **canonical link**. Since
$\mu = b'(\theta)$, it is

$$
g(\mu) = (b')^{-1}(\mu).
$$
:::

If $\phi > 0$, the canonical link is strictly increasing: by
{eq}`eq-glm-variance`, $b''(\theta) = \Var(Y)/\phi > 0$ for any
non-degenerate distribution, so $b'$ is strictly increasing, and so is its
inverse.

For the Bernoulli distribution, $b'(\theta) = e^\theta/(1+e^\theta) = \mu$;
solving for $\theta$ gives the canonical link
$g(\mu) = \ln\{\mu/(1-\mu)\} = \logit(\mu)$, which yields logistic
regression ({doc}`18-logistic-regression`). Other examples:

| Distribution | $b(\theta)$ | Canonical link $g(\mu)$ |
|---|---|---|
| Normal | $\theta^2/2$ | $\mu$ |
| Poisson | $e^\theta$ | $\ln\mu$ |
| Bernoulli | $\ln(1+e^\theta)$ | $\ln\{\mu/(1-\mu)\}$ |
| Gamma | $-\ln(-\theta)$ | $-1/\mu$ |

## The GLM and its likelihood

### Model

Let $(\bx_i, Y_i) \in \R^p\times\R$, $i = 1,\ldots,n$, be independent pairs
such that the conditional distribution of $Y_i$ given $\bx_i$ has density in
the canonical exponential family,

$$
f_{\theta_i}(y_i) = \exp\left[\frac{y_i\theta_i - b(\theta_i)}{\phi} + c(y_i,\phi)\right],
$$

with response vector $\bY = (Y_1,\ldots,Y_n)\T$ and design matrix
$\bX\in\R^{n\times p}$ whose $i$-th row is $\bx_i\T$. The mean
$\mu_i = b'(\theta_i)$ depends on the predictors through the link,

$$
\begin{aligned}
g(\mu_i) &= \eta_i \\
&= \bx_i\T\bbeta .
\end{aligned}
$$

For a given link $g$, the canonical parameter is a function of $\bbeta$:

$$
\begin{aligned}
\theta_i &= (b')^{-1}(\mu_i) \\
&= (b')^{-1}\bigl(g^{-1}(\bx_i\T\bbeta)\bigr) \equiv h(\bx_i\T\bbeta), \\
h &= (b')^{-1}\circ g^{-1} \\
&= (g\circ b')^{-1}.
\end{aligned}
$$

If $g$ is the canonical link, $h$ is the identity and $\theta_i = \eta_i$.

### Log-likelihood

Up to terms that do not depend on $\bbeta$, the log-likelihood is

$$
\begin{aligned}
\ell(\bbeta) &= \sum_{i=1}^n \frac{Y_i\theta_i - b(\theta_i)}{\phi} \\
&= \sum_{i=1}^n \frac{Y_i\,h(\bx_i\T\bbeta) - b\bigl(h(\bx_i\T\bbeta)\bigr)}{\phi}.
\end{aligned}
$$ (eq-glm-loglik)

With the canonical link this simplifies to

$$
\ell(\bbeta) = \sum_{i=1}^n \frac{Y_i\,\bx_i\T\bbeta - b(\bx_i\T\bbeta)}{\phi}.
$$

Its Hessian is $-\phi^{-1}\sum_i b''(\bx_i\T\bbeta)\,\bx_i\bx_i\T$, which is
negative definite when $\phi > 0$ and $\bX$ has full column rank. Hence,
with the canonical link, the log-likelihood is **strictly concave** and the
maximum likelihood estimator, when it exists, is **unique**. With other
links, the log-likelihood need not be concave and may have several local
maxima.

### Deviance

The **saturated model** gives each observation its own mean,
$\widehat{\mu}_i = y_i$, and attains the largest possible log-likelihood
$\ell_{\mathrm{sat}}$. The **deviance** of a fitted model compares its
log-likelihood $\ell(\widehat{\bbeta})$ with that of the saturated model:

$$
D = 2\phi\bigl[\ell_{\mathrm{sat}} - \ell(\widehat{\bbeta})\bigr].
$$ (eq-deviance)

For the normal model, $D = \mathit{SS}_E$. For models with $\phi = 1$, such as
the Poisson, the difference in deviance between two nested models equals
the difference in $-2\ell(\widehat{\bbeta})$, which is the likelihood ratio
statistic used in {doc}`19-poisson-and-count-regression`.

## Estimation algorithms

The likelihood equations $\partial\ell/\partial\bbeta = \bzero$ usually have
no closed-form solution and are solved iteratively. Common methods are:

- the Newton–Raphson method;
- the Fisher scoring method; and
- iteratively reweighted least squares (IRLS).

### Newton–Raphson

Let $\nabla\ell(\bbeta)$ be the gradient (score) and $\mathbf{H}_\ell(\bbeta)$
the Hessian matrix of the log-likelihood. Newton–Raphson updates

$$
\bbeta^{(t+1)} = \bbeta^{(t)} - \mathbf{H}_\ell\bigl(\bbeta^{(t)}\bigr)^{-1}\nabla\ell\bigl(\bbeta^{(t)}\bigr),
$$

which maximizes a quadratic approximation of $\ell$ at each step. The method
is purely numerical; it does not use the fact that the data are random.

### Fisher scoring

Computing the Hessian can be complicated. Fisher scoring replaces it with
its expected value: the **Fisher information**

$$
\mathbf{I}(\bbeta) = -\E\bigl[\mathbf{H}_\ell(\bbeta)\bigr],
$$

which is positive definite and can stand in for $-\mathbf{H}_\ell$ in the
Newton–Raphson update:

$$
\bbeta^{(t+1)} = \bbeta^{(t)} + \mathbf{I}\bigl(\bbeta^{(t)}\bigr)^{-1}\nabla\ell\bigl(\bbeta^{(t)}\bigr).
$$

Fisher scoring has essentially the same convergence properties as
Newton–Raphson, but $\mathbf{I}$ is often easier to compute than
$\mathbf{H}_\ell$. With the canonical link the Hessian does not depend on
$\bY$, so the two methods coincide.

### Iteratively reweighted least squares

For a GLM, the Fisher scoring update can be written as a weighted least
squares problem. Given the current estimate, form the **working response**
and **weights**

$$
\begin{gathered}
z_i = \eta_i + (y_i - \mu_i)\,g'(\mu_i), \\
w_i = \frac{1}{V(\mu_i)\,\{g'(\mu_i)\}^2},
\end{gathered}
$$

evaluated at $\bbeta^{(t)}$, and set $\mathbf{W} = \diag(w_1,\ldots,w_n)$.
Then

$$
\bbeta^{(t+1)} = (\bX\T\mathbf{W}\bX)^{-1}\bX\T\mathbf{W}\mathbf{z}.
$$

Each iteration is a weighted linear regression of $\mathbf{z}$ on $\bX$,
which is why the algorithm is called IRLS. At convergence,
$\phi\,(\bX\T\mathbf{W}\bX)^{-1}$ estimates the covariance matrix of
$\widehat{\bbeta}$.

## Summary and Common Mistakes
### Summary

- A GLM consists of an exponential-family response, a linear predictor
  $\eta = \bx\T\bbeta$, and a link $g(\mu) = \eta$.
- In canonical form, $\E[Y] = b'(\theta)$ and $\Var(Y) = \phi\,b''(\theta)$.
- The canonical link $g = (b')^{-1}$ makes the log-likelihood strictly
  concave.
- The deviance compares a model with the saturated model; differences in
  deviance compare nested models.
- GLMs are fitted by Newton–Raphson, Fisher scoring, or the equivalent IRLS
  algorithm.

### Common mistakes

- Confusing the link function with a transformation of the response. A GLM
  with a log link models $\ln\E[Y]$, not $\E[\ln Y]$.
- Assuming every link yields a concave log-likelihood; only the canonical
  link guarantees this.
- Treating the deviance as $-2\ell(\widehat{\bbeta})$; it is measured
  relative to the saturated model.

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized linear models* (2nd ed.). Chapman & Hall.
- Rigollet, P. (2016). *18.650 Statistics for applications* [Lecture notes]. MIT OpenCourseWare. <https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/>
- Wikipedia contributors. (n.d.). Generalized linear model. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Generalized_linear_model>

# Multinomial Choice Models

## Overview

Multinomial choice models describe how a decision maker chooses one
alternative from a finite set of unordered alternatives, for example a
traveler choosing a travel mode. They are built on **random utility
maximization**: each alternative has a utility with a systematic part that
depends on observed attributes and a random part that the analyst cannot
observe, and the decision maker chooses the alternative with the highest
utility. Different assumptions about the random part give different models;
this chapter derives the multinomial logit (MNL) and multinomial probit
(MNP) models and explains how they are estimated and assessed. It builds on maximum likelihood and the logit link ({doc}`17-generalized-linear-models`) and on the latent-variable view of discrete responses ({doc}`20-ordinal-regression`).

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- express choice probabilities in terms of the joint distribution of the
  random utility components;
- derive the multinomial logit model from i.i.d. Gumbel errors;
- explain the role of the scale parameter and the independence of
  irrelevant alternatives (IIA) property;
- describe the multinomial probit model and why its probabilities require
  simulation; and
- estimate choice models by maximum likelihood and assess their fit with
  the likelihood ratio index.
:::

## Notation

| Symbol | Meaning |
|---|---|
| $C$ | universal choice set (all alternatives for the population) |
| $J$ | number of alternatives in $C$ |
| $C_n$, $J_n$ | feasible choice set of decision maker $n$ and its size, $J_n \le J$ |
| $U_{in}$ | utility of alternative $i$ for decision maker $n$ |
| $V_{in}$ | systematic (deterministic) utility |
| $\varepsilon_{in}$ | random utility component |
| $P_n(i)$ | probability that decision maker $n$ chooses alternative $i$ |
| $N$ | number of decision makers in the sample |

## Random utility framework

### Choice sets

Let $C$ be the **choice set** containing all potential alternatives for some
population and $J$ the number of alternatives. In a travel-mode study, $C$
may contain more than two modes, such as:

- driving alone;
- sharing a ride;
- taxi;
- motorcycle;
- bicycle;
- walking;
- transit bus; and
- rail rapid transit.

For a particular traveler $n$, the feasible choice set $C_n$ may be
considerably smaller; for example, driving alone is infeasible for someone
who does not own a car. Let $J_n \le J$ be the number of feasible
alternatives.

### Choice probabilities

The utility of alternative $i$ for decision maker $n$ is
$U_{in} = V_{in} + \varepsilon_{in}$. Under random utility maximization,

$$
\begin{aligned}
P_n(i) &= \Prob\bigl(U_{in} \ge U_{jn},\ \forall j \in C_n,\ j \ne i\bigr) \\
&= \Prob\bigl(V_{in} + \varepsilon_{in} \ge V_{jn} + \varepsilon_{jn},\ \forall j \in C_n,\ j \ne i\bigr) \\
&= \Prob\bigl(\varepsilon_{jn} \le V_{in} - V_{jn} + \varepsilon_{in},\ \forall j \in C_n,\ j \ne i\bigr).
\end{aligned}
$$ (eq-rum-probability)

Equivalently,

$$
P_n(i) = \Prob\Bigl[V_{in} + \varepsilon_{in} \ge \max_{j\in C_n,\ j\ne i}\bigl(V_{jn} + \varepsilon_{jn}\bigr)\Bigr].
$$

Any particular multinomial choice model follows from
{eq}`eq-rum-probability` once the **joint distribution of the random
terms** is specified. If their joint density is
$f(\varepsilon_{1n}, \ldots, \varepsilon_{J_nn})$ and, without loss of
generality, $i$ is the first alternative in $C_n$, then

$$
P_n(1) = \int_{\varepsilon_{1n}=-\infty}^{\infty}
\int_{\varepsilon_{2n}=-\infty}^{V_{1n}-V_{2n}+\varepsilon_{1n}} \cdots
\int_{\varepsilon_{J_nn}=-\infty}^{V_{1n}-V_{J_nn}+\varepsilon_{1n}}
f(\varepsilon_{1n}, \varepsilon_{2n}, \ldots, \varepsilon_{J_nn})\,
\mathrm{d}\varepsilon_{J_nn}\cdots\mathrm{d}\varepsilon_{2n}\,\mathrm{d}\varepsilon_{1n}.
$$

## Multinomial logit model

### Assumptions

The **multinomial logit (MNL)** model assumes that the random terms
$\varepsilon_{in}$ are:

- independent across alternatives;
- identically distributed; and
- Gumbel distributed with common location $\eta$ and scale parameter
  $\mu > 0$, written $\mathrm{Gumbel}(\eta,\mu)$, with CDF
  $F(\varepsilon) = \exp\{-e^{-\mu(\varepsilon-\eta)}\}$, mean
  $\eta + \gamma/\mu$, and variance $\pi^2/(6\mu^2)$, where
  $\gamma \approx 0.57721$ is the Euler–Mascheroni constant.

The Gumbel assumption can be defended as an approximation to the normal
distribution; it is used mainly for **analytical convenience**. The
assumption that the random terms are independent and identically
distributed (i.i.d.) is, however, a strong restriction:

- the variances of the random components of the utilities are assumed
  equal; and
- the random terms may, in some situations, not be independent.

### Choice probabilities

:::{prf:theorem} Multinomial logit probabilities
:label: thm-mnl

If the $\varepsilon_{in}$, $i\in C_n$, are i.i.d. $\mathrm{Gumbel}(\eta,\mu)$, then

$$
P_n(i) = \frac{\exp(\mu V_{in})}{\sum_{j\in C_n}\exp(\mu V_{jn})},
\qquad i \in C_n .
$$ (eq-mnl)

The scale parameter is related to the common standard deviation $\sigma$ of
the random terms by

$$
\begin{gathered}
\sigma^2 = \frac{\pi^2}{6\mu^2} \quad\text{or} \\
\mu = \frac{\pi}{\sqrt{6}\,\sigma}.
\end{gathered}
$$
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

The location $\eta$ cancels in the differences $\varepsilon_{jn}-\varepsilon_{in}$,
so assume $\eta = 0$: $\varepsilon_{in} \overset{\text{i.i.d.}}{\sim} \mathrm{Gumbel}(0,\mu)$.
Order the alternatives so that $i = 1$ and define

$$
\begin{gathered}
U_n^{*} = \max_{j = 2,\ldots,J_n}\bigl(V_{jn} + \varepsilon_{jn}\bigr), \quad\text{so that} \\
P_n(1) = \Prob\bigl(V_{1n} + \varepsilon_{1n} \ge U_n^{*}\bigr).
\end{gathered}
$$

The maximum of independent Gumbel variables with a common scale is again
Gumbel with that scale:

$$
U_n^{*} \sim \mathrm{Gumbel}\!\left(\frac{1}{\mu}\ln\sum_{j=2}^{J_n}\exp(\mu V_{jn}),\ \mu\right).
$$

Hence $U_n^{*} = V_n^{*} + \varepsilon_n^{*}$, where

$$
\begin{gathered}
V_n^{*} = \frac{1}{\mu}\ln\sum_{j=2}^{J_n}\exp(\mu V_{jn}) \quad\text{and} \\
\varepsilon_n^{*} \sim \mathrm{Gumbel}(0,\mu),
\end{gathered}
$$

independent of $\varepsilon_{1n}$. Then

$$
\begin{aligned}
P_n(1) &= \Prob\bigl(V_{1n} + \varepsilon_{1n} \ge V_n^{*} + \varepsilon_n^{*}\bigr) \\
&= \Prob\bigl(\varepsilon_n^{*} - \varepsilon_{1n} \le V_{1n} - V_n^{*}\bigr).
\end{aligned}
$$

The difference of two independent $\mathrm{Gumbel}(0,\mu)$ variables follows
the logistic distribution with CDF $F(x) = 1/(1+e^{-\mu x})$. Therefore

$$
\begin{aligned}
P_n(1) &= \frac{1}{1 + \exp\bigl[-\mu(V_{1n} - V_n^{*})\bigr]} \\
&= \frac{\exp(\mu V_{1n})}{\exp(\mu V_{1n}) + \exp(\mu V_n^{*})} \\
&= \frac{\exp(\mu V_{1n})}{\sum_{j\in C_n}\exp(\mu V_{jn})},
\end{aligned}
$$

since $\exp(\mu V_n^{*}) = \sum_{j=2}^{J_n}\exp(\mu V_{jn})$.
:::

The scale $\mu$ cannot be identified separately from the coefficients of
$V_{jn}$. With linear-in-parameters utilities
$V_{jn} = \beta_1x_{jn1} + \cdots + \beta_Kx_{jnK} = \bbeta\T\bx_{jn}$,
the model depends only on $\mu V_{jn} = (\mu\bbeta)\T\bx_{jn}$. It is
therefore customary to normalize $\mu = 1$:

$$
P_n(i) = \frac{\exp(V_{in})}{\sum_{j\in C_n}\exp(V_{jn})},
$$

with the understanding that the estimated coefficients are scaled by $\mu$.
With two alternatives, $P_n(1) = 1/\{1 + \exp[-(V_{1n} - V_{2n})]\}$, which
is the binary logistic regression model of {doc}`18-logistic-regression`
applied to the utility difference.

### Extreme values of the scale parameter

Two limiting cases illustrate the role of $\mu = \pi/(\sqrt6\,\sigma)$.

**Case 1: $\mu \to 0$, i.e. $\sigma \to +\infty$.**

$$
\lim_{\mu\to0}P_n(i) = \frac{1}{J_n}, \qquad \forall i\in C_n .
$$

As the variance of the random terms grows without bound, the systematic
utilities carry no information and all feasible alternatives are equally
likely.

**Case 2: $\mu \to +\infty$, i.e. $\sigma \to 0$.**

$$
\begin{aligned}
\lim_{\mu\to+\infty}P_n(i)
&= \lim_{\mu\to+\infty}\frac{1}{1 + \sum_{j\in C_n,\ j\ne i}\exp\bigl[\mu(V_{jn}-V_{in})\bigr]} \\
&= \begin{cases}
1, & \text{if } V_{in} > V_{jn} \text{ for all } j \ne i, \\
0, & \text{if } V_{in} < V_{jn} \text{ for some } j .
\end{cases}
\end{aligned}
$$

As the variance of the random terms approaches zero, the model becomes a
**deterministic choice model**: all information about preferences is in the
systematic utilities, and the alternative with the highest $V_{in}$ is
chosen with certainty.

### Linear-in-parameters logit

Restricting $V_{in}$ to linear-in-parameters functions gives

$$
P_n(i) = \frac{\exp\bigl(\mu\bbeta\T\bx_{in}\bigr)}{\sum_{j\in C_n}\exp\bigl(\mu\bbeta\T\bx_{jn}\bigr)},
$$

where $\bx_{in}$ describes the attributes of alternative $i$ and of decision
maker $n$.

### Independence of irrelevant alternatives

:::{prf:property} Independence of irrelevant alternatives (IIA)
:label: prop-iia

For a given decision maker, the ratio of the MNL choice probabilities of any
two alternatives $i$ and $l$ does not depend on the systematic utilities of
any other alternative:

$$
\begin{aligned}
\frac{P_n(i)}{P_n(l)} &= \frac{\exp(\mu V_{in})\big/\sum_{j\in C_n}\exp(\mu V_{jn})}{\exp(\mu V_{ln})\big/\sum_{j\in C_n}\exp(\mu V_{jn})} \\
&= \exp\bigl[\mu(V_{in}-V_{ln})\bigr].
\end{aligned}
$$
:::

IIA is often seen as a **disadvantage**: it makes the model fail when some
alternatives are close substitutes (**correlated alternatives**). Any model
that assumes mutually independent and identically distributed random terms
yields counter-intuitive probabilities in the classical **red bus/blue bus
problem**: adding a blue bus that is identical to an existing red bus should
split the bus share, but under IIA it draws share proportionally from every
alternative.

## Multinomial probit model

### Assumption

The **multinomial probit (MNP)** model assumes jointly normal random terms,

$$
\bvarepsilon_n = (\varepsilon_{1n}, \varepsilon_{2n}, \ldots, \varepsilon_{J_nn})\T \sim \mathcal{N}(\bzero, \bSigma),
$$

with mean $\E[\varepsilon_{in}] = 0$ for $i\in C_n$ and a general
$J_n\times J_n$ covariance matrix

$$
\bSigma = \Bigl[\E\bigl[(\varepsilon_{in}-\E[\varepsilon_{in}])(\varepsilon_{jn}-\E[\varepsilon_{jn}])\bigr]\Bigr]_{J_n\times J_n},
$$

which allows unequal variances and correlation between alternatives.

### Choice probabilities

Let $\mathbf{u} = (u_1,\ldots,u_{J_n})\T$ denote a realization of the utility
vector $\mathbf{U}_n = \mathbf{V}_n + \bvarepsilon_n$, where
$\mathbf{V}_n = (V_{1n},\ldots,V_{J_nn})\T$. Then
$\mathbf{U}_n \sim \mathcal{N}(\mathbf{V}_n, \bSigma)$ and

$$
\begin{aligned}
P_n(i) &= \Prob\bigl(U_{in} \ge U_{jn},\ \forall j\ne i\bigr) \\
&= \int_{\{\mathbf{u}:\ u_i \ge u_j\ \forall j\ne i\}} (2\pi)^{-J_n/2}\,\lvert\bSigma\rvert^{-1/2} \exp\Bigl[-\tfrac12(\mathbf{u}-\mathbf{V}_n)\T\bSigma^{-1}(\mathbf{u}-\mathbf{V}_n)\Bigr] \,\mathrm{d}\mathbf{u}.
\end{aligned}
$$

- The choice probability has no closed form.
- In the binary case, it reduces to a univariate normal CDF and is easy to
  compute.
- In general, it is evaluated by numerical approximation or **Monte Carlo
  simulation**.

## Specifying the systematic utility

- A variable that does **not vary across alternatives**, such as the
  decision maker's income, cancels from the utility differences if it
  enters every utility with the same coefficient. It can therefore be
  included, with alternative-specific coefficients, in at most $J-1$ of the
  utility functions.
- Similarly, a categorical variable with $L$ categories is encoded with at
  most $L-1$ dummy variables.
- Categorizing a continuous variable is usually inadvisable.

## Maximum likelihood estimation

### Likelihood

For a sample of $N$ decision makers, let

$$
y_{n,i} = \begin{cases} 1, & \text{if decision maker } n \text{ chooses alternative } i, \\ 0, & \text{otherwise.} \end{cases}
$$

The likelihood of the coefficients $\bbeta = (\beta_1,\ldots,\beta_K)\T$ is

$$
\mathcal{L}(\bbeta) = \prod_{n=1}^N\prod_{i\in C_n}P_n(i)^{\,y_{n,i}},
$$

and the log-likelihood is

$$
\begin{aligned}
\ell(\bbeta) &= \ln\mathcal{L}(\bbeta) \\
&= \sum_{n=1}^N\sum_{i\in C_n}y_{n,i}\ln P_n(i).
\end{aligned}
$$

### Estimation

The maximum likelihood estimator $\widehat{\bbeta}$ is the value of $\bbeta$
under which the observed choices are most probable:

$$
\begin{aligned}
\widehat{\bbeta} &= \operatorname*{arg\,max}_{\bbeta}\mathcal{L}(\bbeta) \\
&= \operatorname*{arg\,max}_{\bbeta}\ell(\bbeta).
\end{aligned}
$$

For the MNL model with linear-in-parameters utilities, $\ell$ is concave, so
numerical maximization is reliable.

### Properties of the log-likelihood and goodness of fit

Because each $P_n(i) \le 1$, the log-likelihood is non-positive.

- $\ell = 0$ corresponds to $\mathcal{L} = 1$: every choice is predicted with
  probability one (perfect prediction).
- $\ell(\bzero)$ is the log-likelihood with all coefficients equal to zero;
  then all feasible alternatives are equally likely and
  $\ell(\bzero) = -\sum_n \ln J_n$.

The **likelihood ratio index** measures the fraction of the initial
log-likelihood explained by the model:

$$
\rho^2 = 1 - \frac{\ell(\widehat{\bbeta})}{\ell(\bzero)}.
$$

- $\rho^2$ plays a role analogous to $R^2$ in linear regression.
- $0 \le \rho^2 \le 1$.
- $\rho^2$ never decreases when parameters are added. The **adjusted**
  index $\bar\rho^2 = 1 - \bigl(\ell(\widehat{\bbeta}) - K\bigr)/\ell(\bzero)$
  penalizes the number of parameters $K$.

### Principles for selecting variables

When deciding whether to keep a variable, consider:

- the **signs and relative magnitudes** of the estimated coefficients; for
  example, the coefficient of price is expected to be negative;
- **statistical tests** of the significance of each coefficient; and
- **goodness-of-fit** measures.

The following table summarizes a common decision rule, distinguishing
policy variables (those the analysis is meant to evaluate) from other
variables:

| Sign | Significance | Policy variable | Other variable |
|---|---|---|---|
| Correct | Significant | Include | Include |
| Correct | Not significant | Include | May reject |
| Wrong | Significant | Big problem | Reject |
| Wrong | Not significant | Problem | Reject |

## Software

- **Apollo** (R): a free package for estimating a wide range of choice
  models. It does not rely on commercial software and runs wherever R runs.
  <http://www.apollochoicemodelling.com/manual.html>
- **mlogit** (R): estimation of multinomial logit models; see the package
  vignette "Estimation of multinomial logit models in R: The mlogit
  packages". <https://cran.r-project.org/package=mlogit>
- **Biogeme** (Python): maximum likelihood estimation of parametric choice
  models. <https://biogeme.epfl.ch/>

## Summary and Common Mistakes
### Summary

- Choice probabilities follow from random utility maximization once the
  joint distribution of the random terms is specified.
- i.i.d. Gumbel errors give the MNL model
  $P_n(i) = e^{\mu V_{in}}/\sum_{j}e^{\mu V_{jn}}$; the scale $\mu$ is
  normalized to one.
- MNL implies IIA, which fails for correlated alternatives; MNP allows a
  general covariance matrix but needs simulation.
- Choice models are estimated by maximum likelihood; $\rho^2$ and
  $\bar\rho^2$ measure fit.

### Common mistakes

- Including an alternative-invariant variable with the same coefficient in
  every utility; it cancels and its coefficient is not identified.
- Comparing MNL coefficients across data sets without accounting for the
  scale normalization $\mu = 1$.
- Using MNL when alternatives are close substitutes, where IIA is
  implausible.

## References

- Ben-Akiva, M., & Lerman, S. R. (1985). *Discrete choice analysis: Theory and application to travel demand*. MIT Press.
- Train, K. E. (2009). *Discrete choice methods with simulation* (2nd ed.). Cambridge University Press. <https://doi.org/10.1017/CBO9780511805271>
- Bierlaire, M. (n.d.). *Biogeme* [Computer software]. EPFL. <https://biogeme.epfl.ch/>
- Croissant, Y. (n.d.). *mlogit: Multinomial logit models* (R package) [Computer software]. <https://CRAN.R-project.org/package=mlogit>
- Hess, S., & Palma, D. (n.d.). *Apollo: Choice modelling in R* [Computer software]. <http://www.apollochoicemodelling.com/>

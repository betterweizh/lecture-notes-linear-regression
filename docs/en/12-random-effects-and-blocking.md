# Random Effects and Blocking

## Overview

{doc}`11-one-way-anova` treated the $a$ treatments as fixed levels chosen by
the experimenter. This chapter covers two extensions of the single-factor
experiment. In the **random-effects model**, the treatments are a random
sample from a larger population of levels, and the question is how much of
the variability in the response is due to the factor. In the **randomized
complete block design (RCBD)**, a nuisance factor that would otherwise
inflate the error is controlled by testing every treatment within each
block. Both use the same sums of squares as one-way ANOVA but lead to
different hypotheses and expected mean squares. It builds on sums of squares, the $F$-test, and LSD ({doc}`11-one-way-anova`) and on the variance of sums of independent variables ({doc}`01-probability-and-random-vectors`).

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- distinguish fixed and random factors, and state the random-effects model
  and its variance components;
- derive the expected mean squares of the random-effects model and test
  $H_0:\sigma_\tau^2 = 0$;
- estimate the variance components, including for unequal group sizes;
- explain when and why to block on a nuisance factor; and
- analyze a randomized complete block design with its ANOVA table and
  compare treatment means with Fisher's LSD.
:::

## Notation

| Symbol | Meaning |
|---|---|
| $a$, $n$, $N = an$ | number of treatments, replicates per treatment, observations |
| $\tau_i$ | effect of treatment $i$ (random in the random-effects model) |
| $\sigma^2$, $\sigma_\tau^2$ | error variance and treatment variance component |
| $b$ | number of blocks in a randomized complete block design |
| $\beta_j$ | effect of block $j$ |
| $\bar{y}_{i\cdot}$, $\bar{y}_{\cdot j}$, $\bar{y}_{\cdot\cdot}$ | treatment mean, block mean, grand mean |

## The random-effects model

### Fixed and random factors

A factor is **fixed** when its levels are chosen deliberately and
inference is confined to those levels; quantitative factors such as
temperature are usually fixed. A factor is **random** when its levels are a
random sample from a population of possible levels (batches of raw
material, operators, machines) and inference concerns the whole population.
Measurement-system studies in industry are a typical application.

:::{prf:assumption} One-way random-effects model
:label: asm-random-effects

$$
y_{ij} = \mu + \tau_i + \varepsilon_{ij},
\qquad i = 1,\ldots,a,\quad j = 1,\ldots,n,
$$

where $\tau_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_\tau^2)$,
$\varepsilon_{ij} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$, and
the $\tau_i$ are independent of the $\varepsilon_{ij}$.
:::

The variance of an observation is the sum of two **variance components**,

$$
\Var(y_{ij}) = \sigma_\tau^2 + \sigma^2 ,
$$

and two observations under the same treatment are correlated, with
$\Cov(y_{ij}, y_{ij'}) = \sigma_\tau^2$ for $j \ne j'$. The ratio
$\sigma_\tau^2/(\sigma_\tau^2 + \sigma^2)$ is the **intraclass correlation**.

### Hypotheses

Testing the equality of the particular treatment means is no longer of
interest: the treatments in the experiment are a sample, and the question
is whether the population of treatments varies at all.

$$
\begin{gathered}
H_0:\ \sigma_\tau^2 = 0 \quad\text{versus} \\
H_1:\ \sigma_\tau^2 > 0 .
\end{gathered}
$$

### Expected mean squares and the test

The sums of squares, degrees of freedom, and mean squares are computed
exactly as in {doc}`11-one-way-anova`, and the identity
$\mathit{SS}_T = \mathit{SS}_{\mathrm{Treatments}} + \mathit{SS}_E$ still
holds. Only the expected mean squares change.

:::{prf:theorem} Expected mean squares in the random-effects model
:label: thm-random-ems

Under {prf:ref}`asm-random-effects`,

$$
\begin{gathered}
\E[\mathit{MS}_E] = \sigma^2, \\
\E[\mathit{MS}_{\mathrm{Treatments}}] = \sigma^2 + n\,\sigma_\tau^2 .
\end{gathered}
$$
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

Within treatment $i$, $y_{ij} - \bar{y}_{i\cdot} = \varepsilon_{ij} - \bar{\varepsilon}_{i\cdot}$
does not involve $\tau_i$, so $\mathit{SS}_E$ has the same distribution as
in the fixed-effects model and $\E[\mathit{MS}_E] = \sigma^2$
({prf:ref}`thm-one-way-ems`). The treatment means
$\bar{y}_{i\cdot} = \mu + \tau_i + \bar{\varepsilon}_{i\cdot}$ are i.i.d.
with variance $\sigma_\tau^2 + \sigma^2/n$, and
$\mathit{MS}_{\mathrm{Treatments}} = n\sum_i(\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot})^2/(a-1)$
is $n$ times their sample variance. A sample variance is unbiased for the
variance ({prf:ref}`ex-sample-variance`), so
$\E[\mathit{MS}_{\mathrm{Treatments}}] = n(\sigma_\tau^2 + \sigma^2/n)$.
:::

Under $H_0$ both mean squares estimate $\sigma^2$; the treatment means are
then i.i.d. normal and, as in {prf:ref}`thm-one-way-distributions`,

$$
F_0 = \frac{\mathit{MS}_{\mathrm{Treatments}}}{\mathit{MS}_E} \sim F(a-1,\ N-a).
$$

$H_0$ is rejected when $F_0 > F_{1-\alpha;\,a-1,\,N-a}$. The test statistic
is the same as in the fixed-effects model; what differs is the hypothesis
and the scope of the conclusion.

### Estimating the variance components

Equating the mean squares to their expectations gives the **analysis of
variance estimators**

$$
\begin{gathered}
\widehat{\sigma}^2 = \mathit{MS}_E, \\
\widehat{\sigma}_\tau^2 = \frac{\mathit{MS}_{\mathrm{Treatments}} - \mathit{MS}_E}{n} .
\end{gathered}
$$

Both are unbiased. The estimate $\widehat{\sigma}_\tau^2$ can be negative
when $\mathit{MS}_{\mathrm{Treatments}} < \mathit{MS}_E$; a negative value
is usually reported as zero and suggests that the treatment variability is
small, or that the model is not appropriate.

For **unequal group sizes** $n_1,\ldots,n_a$ with $N = \sum_i n_i$, replace
$n$ in $\E[\mathit{MS}_{\mathrm{Treatments}}]$ and in $\widehat{\sigma}_\tau^2$ by

$$
n_0 = \frac{1}{a-1}\left(N - \frac{\sum_{i=1}^a n_i^2}{N}\right).
$$

When all $n_i = n$, $n_0 = n$.

## Blocking

### Nuisance factors

A **nuisance factor** probably affects the response but is not of interest
in itself: batches of raw material, operators, pieces of test equipment,
days or shifts, or different experimental units. Its variability should be
kept out of the comparison of treatments.

- If the nuisance factor is **known and controllable**, use **blocking**.
- If it is **known but uncontrollable** and can be measured, the **analysis
  of covariance** can remove its effect.
- If it is **unknown and uncontrollable** (a lurking variable),
  **randomization** balances its effect across the treatments on average.

A **block** is one level of the nuisance factor, a set of experimental
units that are more homogeneous than the units as a whole. Variability
between blocks may be large, but variability within a block should be
small. Several nuisance sources can be combined into one block variable.

### The randomized complete block design

In a **randomized complete block design (RCBD)**, each block contains every
treatment exactly once ("complete"), and the order of the runs is
randomized separately within each block. For example, to compare $a = 4$
tips of a hardness-testing machine, each tip is tested once on each of
$b = 4$ metal coupons; the coupons are the blocks, so differences between
coupons are removed from the experimental error. The block is a
**restriction on randomization**: runs are randomized within, not across,
blocks.

:::{prf:assumption} Randomized complete block design
:label: asm-rcbd

$$
y_{ij} = \mu + \tau_i + \beta_j + \varepsilon_{ij},
\qquad i = 1,\ldots,a,\quad j = 1,\ldots,b,
$$

where the treatment and block effects are fixed with
$\sum_i\tau_i = 0$ and $\sum_j\beta_j = 0$, and
$\varepsilon_{ij} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$.
:::

The treatment means are
$\mu_i = \frac1b\sum_j(\mu + \tau_i + \beta_j) = \mu + \tau_i$, and the
hypotheses of interest are

$$
\begin{gathered}
H_0:\ \mu_1 = \mu_2 = \cdots = \mu_a \quad\text{versus} \\
H_1:\ \mu_i \ne \mu_j \text{ for at least one pair } (i,j).
\end{gathered}
$$

### Partition of the total variability

With treatment means $\bar{y}_{i\cdot} = \frac1b\sum_j y_{ij}$, block means
$\bar{y}_{\cdot j} = \frac1a\sum_i y_{ij}$, and grand mean
$\bar{y}_{\cdot\cdot} = \frac{1}{ab}\sum_i\sum_j y_{ij}$, write

$$
y_{ij} - \bar{y}_{\cdot\cdot}
= (\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot})
+ (\bar{y}_{\cdot j} - \bar{y}_{\cdot\cdot})
+ (y_{ij} - \bar{y}_{i\cdot} - \bar{y}_{\cdot j} + \bar{y}_{\cdot\cdot}) .
$$

Squaring and summing over $i$ and $j$, the cross terms vanish because each
component sums to zero over the index it does not share with the other, so

$$
\begin{aligned}
\sum_{i=1}^a\sum_{j=1}^b(y_{ij} - \bar{y}_{\cdot\cdot})^2
&= \underbrace{b\sum_{i=1}^a(\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot})^2}_{\mathit{SS}_{\mathrm{Treatments}}}
 + \underbrace{a\sum_{j=1}^b(\bar{y}_{\cdot j} - \bar{y}_{\cdot\cdot})^2}_{\mathit{SS}_{\mathrm{Blocks}}} \\
&\quad + \underbrace{\sum_{i=1}^a\sum_{j=1}^b(y_{ij} - \bar{y}_{i\cdot} - \bar{y}_{\cdot j} + \bar{y}_{\cdot\cdot})^2}_{\mathit{SS}_E},
\end{aligned}
$$

that is,
$\mathit{SS}_T = \mathit{SS}_{\mathrm{Treatments}} + \mathit{SS}_{\mathrm{Blocks}} + \mathit{SS}_E$,
with degrees of freedom

$$
ab - 1 = (a-1) + (b-1) + (a-1)(b-1).
$$

### Expected mean squares and the $F$-test

Dividing each sum of squares by its degrees of freedom gives the mean
squares, with expectations

$$
\begin{gathered}
\E[\mathit{MS}_{\mathrm{Treatments}}] = \sigma^2 + \frac{b}{a-1}\sum_{i=1}^a\tau_i^2, \\
\E[\mathit{MS}_{\mathrm{Blocks}}] = \sigma^2 + \frac{a}{b-1}\sum_{j=1}^b\beta_j^2, \\
\E[\mathit{MS}_E] = \sigma^2 .
\end{gathered}
$$

Under $H_0$ the treatment mean square also estimates $\sigma^2$, and

$$
F_0 = \frac{\mathit{MS}_{\mathrm{Treatments}}}{\mathit{MS}_E} \sim F\bigl(a-1,\ (a-1)(b-1)\bigr).
$$

$H_0$ is rejected when $F_0 > F_{1-\alpha;\,a-1,\,(a-1)(b-1)}$.

| Source of variation | Sum of squares | Degrees of freedom | Mean square | $F_0$ |
|---|---|---|---|---|
| Treatments | $\mathit{SS}_{\mathrm{Treatments}}$ | $a-1$ | $\mathit{SS}_{\mathrm{Treatments}}/(a-1)$ | $\mathit{MS}_{\mathrm{Treatments}}/\mathit{MS}_E$ |
| Blocks | $\mathit{SS}_{\mathrm{Blocks}}$ | $b-1$ | $\mathit{SS}_{\mathrm{Blocks}}/(b-1)$ | |
| Error | $\mathit{SS}_E$ (by subtraction) | $(a-1)(b-1)$ | $\mathit{SS}_E/\bigl((a-1)(b-1)\bigr)$ | |
| Total | $\mathit{SS}_T$ | $ab-1$ | | |

Computing formulas, with $N = ab$, are

$$
\begin{gathered}
\mathit{SS}_T = \sum_{i=1}^a\sum_{j=1}^b y_{ij}^2 - \frac{y_{\cdot\cdot}^2}{N}, \\
\mathit{SS}_{\mathrm{Treatments}} = \frac1b\sum_{i=1}^a y_{i\cdot}^2 - \frac{y_{\cdot\cdot}^2}{N}, \\
\mathit{SS}_{\mathrm{Blocks}} = \frac1a\sum_{j=1}^b y_{\cdot j}^2 - \frac{y_{\cdot\cdot}^2}{N}, \\
\mathit{SS}_E = \mathit{SS}_T - \mathit{SS}_{\mathrm{Treatments}} - \mathit{SS}_{\mathrm{Blocks}} .
\end{gathered}
$$

Blocks are a restriction on randomization, so the ratio
$\mathit{MS}_{\mathrm{Blocks}}/\mathit{MS}_E$ is not an exact test of block
effects; a large value simply confirms that blocking was worthwhile.

### Multiple comparisons

When $H_0$ is rejected, pairs of treatment means are compared with Fisher's
LSD ({doc}`11-one-way-anova`), using $b$ observations per treatment and the
error degrees of freedom of the RCBD:

$$
\mathrm{LSD} = t_{1-\alpha/2,\,(a-1)(b-1)}\sqrt{\frac{2\,\mathit{MS}_E}{b}} .
$$

The pair $\mu_i$, $\mu_j$ is declared different when
$\lvert\bar{y}_{i\cdot} - \bar{y}_{j\cdot}\rvert > \mathrm{LSD}$.

## Summary and Common Mistakes
### Summary

- In the random-effects model, $\Var(y_{ij}) = \sigma_\tau^2 + \sigma^2$;
  $\E[\mathit{MS}_{\mathrm{Treatments}}] = \sigma^2 + n\sigma_\tau^2$, and
  $F_0 = \mathit{MS}_{\mathrm{Treatments}}/\mathit{MS}_E$ tests
  $H_0:\sigma_\tau^2 = 0$.
- $\widehat{\sigma}^2 = \mathit{MS}_E$ and
  $\widehat{\sigma}_\tau^2 = (\mathit{MS}_{\mathrm{Treatments}} - \mathit{MS}_E)/n$.
- Blocking removes the variability of a known, controllable nuisance factor
  from the error.
- In an RCBD,
  $\mathit{SS}_T = \mathit{SS}_{\mathrm{Treatments}} + \mathit{SS}_{\mathrm{Blocks}} + \mathit{SS}_E$,
  and treatments are tested with $F(a-1,(a-1)(b-1))$.

### Common mistakes

- Testing the equality of particular treatment means when the treatments
  are a random sample; the hypothesis concerns $\sigma_\tau^2$.
- Reporting a negative estimate of $\sigma_\tau^2$ without comment.
- Ignoring a known nuisance factor; its variability then inflates
  $\mathit{MS}_E$ and reduces the power of the treatment test.
- Analyzing an RCBD as a one-way ANOVA, which puts the block variability
  into the error.

## References

- Montgomery, D. C. (2017). *Design and analysis of experiments* (9th ed.). Wiley.
- Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley.

# Variable and Subset Selection

## Overview

With many candidate predictors, the analyst must decide which ones to keep.
A model with more predictors may describe the data in more detail but is
harder to interpret and may overfit; a model with fewer predictors is
simpler to use but may omit important effects. This chapter collects the
criteria used to compare candidate models and the search algorithms used to
find good subsets, and shows how to run them in R. It builds on the partial $F$-test ({doc}`10-anova-and-inference`) and on $R^2$, adjusted $R^2$, and PRESS ({doc}`13-prediction-and-model-assessment`). The next chapter, {doc}`16-ridge-and-lasso`, presents penalized regression as a continuous alternative to discrete subset selection.

:::{admonition} Learning objectives
:class: objectives
After completing this chapter, readers should be able to:

- state the variable selection problem and its assumptions;
- compute and compare $R^2_p$, $R^2_{\mathrm{adj}}(p)$, $\mathit{MS}_E(p)$,
  Mallows' $C_p$, AIC, AICc, and BIC; and
- describe best subset, forward, backward, and stepwise selection and their
  computational costs.
:::

## The selection problem

### Motivation

- In model building, it is important to select relevant predictors from a
  set of possible ones.
- Selection involves screening the candidates to obtain a "best" subset.
- There is a trade-off between a detailed model with many predictors and a
  general model with few predictors that is easy to use.
- There is no single "best" model: different criteria and procedures can
  select different subsets. Overfitting should be avoided, and subject-matter
  judgment is still required.

### Problem statement

Choose the "best" subset of predictors from the candidate set
$\{x_1, x_2, \ldots, x_K\}$, under the following assumptions:

- there is one response $y$ and $K$ candidate predictors;
- every model contains the intercept $\beta_0$; and
- a subset model has $p$ coefficients, i.e. the intercept and $p-1$
  predictors, with $p = 1, 2, \ldots, K+1$.

Two questions must be answered:

1. **a criterion** that measures the quality of a subset; and
2. **a search method** that finds a good subset without necessarily fitting
   all of them.

In what follows, $\mathit{SS}_E(p)$, $\mathit{SS}_R(p)$, and $R_p^2$ denote
quantities computed from a subset model with $p$ coefficients, and
$\mathit{SS}_T$ does not depend on the subset.

## Selection criteria

### $R^2$ criterion

$$
\begin{aligned}
R_p^2 &= \frac{\mathit{SS}_R(p)}{\mathit{SS}_T} \\
&= 1 - \frac{\mathit{SS}_E(p)}{\mathit{SS}_T}.
\end{aligned}
$$

As $p$ increases, $\mathit{SS}_T$ stays constant, $\mathit{SS}_R(p)$
increases, and $\mathit{SS}_E(p)$ decreases, so the largest model always has
the largest $R_p^2$. The criterion is therefore used graphically: sort the
models by $p$, plot the **upper envelope** of $R_p^2$ against $p$, and choose
the point beyond which adding predictors yields little improvement.

### Adjusted $R^2$ criterion

$$
R^2_{\mathrm{adj}}(p) = 1 - \frac{n-1}{n-p}\bigl(1-R_p^2\bigr).
$$

As $p$ increases, both $1-R_p^2$ and $n-p$ decrease, so
$R^2_{\mathrm{adj}}(p)$ does not necessarily increase; whether it does
depends on which effect dominates. **Criterion:** select the model with the
largest $R^2_{\mathrm{adj}}(p)$.

### Residual mean square criterion

$$
\mathit{MS}_E(p) = \frac{\mathit{SS}_E(p)}{n-p}.
$$

As $p$ increases, both $\mathit{SS}_E(p)$ and $n-p$ decrease. Typically the
lower envelope of $\mathit{MS}_E(p)$ decreases as $p$ increases, reaches a
minimum, and then may increase. **Criterion:** select the model with the
smallest $\mathit{MS}_E(p)$.

:::{prf:proposition} Equivalence of the $\mathit{MS}_E$ and adjusted $R^2$ criteria
:label: prop-mse-adjr2

The subset model that minimizes $\mathit{MS}_E(p)$ also maximizes
$R^2_{\mathrm{adj}}(p)$.
:::

:::{dropdown} Proof
:class-container: proof
:animate: fade-in

$$
\begin{aligned}
R^2_{\mathrm{adj}}(p) &= 1 - \frac{n-1}{n-p}\cdot\frac{\mathit{SS}_E(p)}{\mathit{SS}_T} \\
&= 1 - \frac{n-1}{\mathit{SS}_T}\cdot\frac{\mathit{SS}_E(p)}{n-p} \\
&= 1 - \frac{n-1}{\mathit{SS}_T}\,\mathit{MS}_E(p).
\end{aligned}
$$

Since $(n-1)/\mathit{SS}_T$ is a positive constant that does not depend on
the subset, $R^2_{\mathrm{adj}}(p)$ is a decreasing function of
$\mathit{MS}_E(p)$.
:::

### Mallows' $C_p$

**Mallows' $C_p$** is defined as

$$
C_p = \frac{\mathit{SS}_E(p)}{\widehat{\sigma}^2} - n + 2p,
$$ (eq-mallows-cp)

where $\mathit{SS}_E(p)$ is the residual sum of squares of the subset model
with $p$ coefficients and $\widehat{\sigma}^2$ is estimated from the **full
model** containing all $K$ candidate predictors. An alternative version is

$$
\begin{aligned}
C_p^{*} &= \frac{1}{n}\bigl(\mathit{SS}_E(p) + 2p\,\widehat{\sigma}^2\bigr) \\
&= \frac{\widehat{\sigma}^2}{n}\bigl(C_p + n\bigr).
\end{aligned}
$$

Both versions are minimized by the same subset. A subset model with little
bias has $C_p \approx p$.

### Akaike information criterion (AIC)

#### General definition

The **Akaike information criterion** of a model with $d$ estimated
parameters and maximized likelihood $\widehat{\mathcal{L}}$ is

$$
\mathrm{AIC} = 2d - 2\ln\widehat{\mathcal{L}} .
$$

Smaller values are better.

#### AIC in linear regression

For a normal linear model, substituting the ML estimators of
{doc}`08-ols-and-ml-estimation` into the log-likelihood {eq}`eq-normal-loglik`, with
$\widehat{\sigma}^2_{\mathrm{ML}} = \mathit{SS}_E/n$, gives

$$
\begin{aligned}
\ln\widehat{\mathcal{L}}
&= -\frac{n}{2}\ln\bigl(2\pi\widehat{\sigma}^2_{\mathrm{ML}}\bigr)
   - \frac{1}{2\widehat{\sigma}^2_{\mathrm{ML}}}\sum_{i=1}^n(y_i-\widehat{y}_i)^2 \\
 &= -\frac{n}{2}\ln\bigl(2\pi\widehat{\sigma}^2_{\mathrm{ML}}\bigr) - \frac{n}{2} \\
&= -\frac{n}{2}\ln\!\left(\frac{\mathit{SS}_E}{n}\right)
   \underbrace{-\ \frac{n}{2}\ln(2\pi) - \frac{n}{2}}_{\text{constant}} .
\end{aligned}
$$

A linear model with $p$ coefficients has $d = p + 1$ parameters (the
coefficients and $\sigma^2$). Dropping the terms that are the same for all
models fitted to the same data gives the commonly used form

$$
\mathrm{AIC} = n\ln\!\left(\frac{\mathit{SS}_E}{n}\right) + 2p .
$$ (eq-aic-linear)

Counting $d = p$ or $d = p+1$ changes AIC by the same constant for every
model and does not affect the ranking.

:::{note}
Software may report AIC values that include the constant terms. For
example, `statsmodels` computes AIC from the full log-likelihood
`res.llf`, so its values differ from {eq}`eq-aic-linear` by a constant.
Differences in AIC between models fitted to the same data are unaffected.
:::

#### AIC and Mallows' $C_p$

If the error variance is fixed at the full-model estimate
$\widehat{\sigma}^2$ instead of being estimated separately for each subset,
then $-2\ln\widehat{\mathcal{L}} = n\ln(2\pi\widehat{\sigma}^2) + \mathit{SS}_E/\widehat{\sigma}^2$
and

$$
\begin{aligned}
\mathrm{AIC}
&= 2p + n\ln(2\pi\widehat{\sigma}^2) + \frac{\mathit{SS}_E(p)}{\widehat{\sigma}^2} \\
&= n\ln(2\pi\widehat{\sigma}^2) + C_p + n \\
 &= n\ln(2\pi\widehat{\sigma}^2) + \frac{n}{\widehat{\sigma}^2}\,C_p^{*} .
\end{aligned}
$$

In this case AIC and Mallows' $C_p$ select the same model.

#### Corrected AIC

For small samples, the **corrected AIC** (Hurvich & Tsai, 1989) adds a
second-order bias correction:

$$
\mathrm{AICc} = \mathrm{AIC} + \frac{2d^2 + 2d}{n - d - 1}.
$$

AICc approaches AIC as $n\to\infty$.

### Bayesian information criterion (BIC)

The **Bayesian information criterion** is

$$
\mathrm{BIC} = d\ln(n) - 2\ln\widehat{\mathcal{L}},
$$

where $d$ is the number of estimated parameters and $n$ the number of
observations. For linear regression, up to an additive constant,

$$
\mathrm{BIC} = n\ln\!\left(\frac{\mathit{SS}_E}{n}\right) + p\ln(n).
$$

Because $\ln(n) > 2$ for $n \ge 8$, BIC penalizes additional coefficients
more heavily than AIC and tends to select smaller models.

### Summary of criteria

| Criterion | Best model | Penalizes model size? |
|---|---|---|
| $R_p^2$ | largest (use the envelope plot) | no |
| $R^2_{\mathrm{adj}}(p)$ | largest | yes |
| $\mathit{MS}_E(p)$ | smallest | yes (equivalent to $R^2_{\mathrm{adj}}$) |
| $R^2_{\mathrm{pred}}$ / PRESS | largest / smallest | implicitly, through cross-validation |
| Mallows' $C_p$ | smallest, with $C_p \approx p$ | yes |
| AIC, AICc | smallest | yes |
| BIC | smallest | yes, more strongly |

## Search algorithms

### Best subset selection

Best subset selection fits every possible combination of the $K$
candidates, i.e. $2^K$ models. For example, with $K = 2$ candidates $x_1$ and
$x_2$ there are $2^2 = 4$ models.

:::{prf:algorithm} Best subset selection
:label: alg-best-subset

1. Let $\mathcal{M}_0$ denote the *null model*, which contains only the
   intercept. It predicts the sample mean $\bar{y}$ for every observation.
2. For $m = 1, 2, \ldots, K$:
   1. fit all $\binom{K}{m}$ models that contain exactly $m$ predictors;
   2. among them, pick the one with the smallest $\mathit{SS}_E$
      (equivalently, the largest $R^2$) and call it $\mathcal{M}_m$.
3. Select a single model from $\mathcal{M}_0, \mathcal{M}_1, \ldots, \mathcal{M}_K$
   using cross-validated prediction error or a criterion that accounts for
   model size, such as $C_p$, AIC, BIC, adjusted $R^2$, or predicted $R^2$.
:::

In step 3, $\mathit{SS}_E$ and $R^2$ cannot be used: as the number of
predictors increases, $\mathit{SS}_E$ of $\mathcal{M}_0,\ldots,\mathcal{M}_K$
decreases monotonically and $R^2$ increases monotonically.

### Forward stepwise selection

:::{prf:algorithm} Forward stepwise selection
:label: alg-forward

1. Start from the null model $\mathcal{M}_0$.
2. For $m = 0, 1, \ldots, K-1$:
   1. consider all $K-m$ models that add one predictor to those in
      $\mathcal{M}_m$;
   2. choose the best of them (smallest $\mathit{SS}_E$, or largest $R^2$)
      and call it $\mathcal{M}_{m+1}$.
3. Select a single model from $\mathcal{M}_0, \ldots, \mathcal{M}_K$ as in
   step 3 of {prf:ref}`alg-best-subset`.
:::

Forward selection fits $1 + K(K+1)/2$ models instead of $2^K$.

### Backward stepwise selection

:::{prf:algorithm} Backward stepwise selection
:label: alg-backward

1. Start from the full model $\mathcal{M}_K$ containing all $K$ predictors.
2. For $m = K, K-1, \ldots, 1$:
   1. consider all $m$ models that remove one predictor from
      $\mathcal{M}_m$, each containing $m-1$ predictors;
   2. choose the best of them (smallest $\mathit{SS}_E$, or largest $R^2$)
      and call it $\mathcal{M}_{m-1}$.
3. Select a single model from $\mathcal{M}_0, \ldots, \mathcal{M}_K$ as in
   step 3 of {prf:ref}`alg-best-subset`.
:::

Backward selection requires $n > K$ so that the full model can be fitted.

### Hybrid (stepwise) selection

**Stepwise selection** (hybrid selection) combines the two directions. It starts with no predictors and adds the most useful
predictor at each step, as in forward selection. After each addition, it
removes any predictor that no longer improves the fit, as in backward
selection. This mimics best subset selection more closely while keeping
the computational advantages of the stepwise methods.

#### Stepwise selection based on partial $F$-tests

A classical version of stepwise regression, probably the most widely used
selection technique, makes each decision with the partial $F$-test of
{doc}`10-anova-and-inference`:

- two thresholds are chosen: $\alpha_{\text{in}}$ for entering and
  $\alpha_{\text{out}}$ for removing, with
  $\alpha_{\text{in}} \le \alpha_{\text{out}}$ so that a predictor that has
  just entered is not removed immediately;
- a predictor $x_j$ is **added** if the $p$-value of its partial $F$
  statistic is below $\alpha_{\text{in}}$; and
- a predictor $x_j$ is **removed** if the $p$-value of its partial $F$
  statistic is above $\alpha_{\text{out}}$.

:::{prf:algorithm} Stepwise selection with partial $F$-tests
:label: alg-stepwise-f

1. **Start.** Enter the single predictor most correlated with $y$, say
   $x_1$.
2. **Add.** For each remaining candidate $x_j$, $j = 2,\ldots,K$, compute

   $$
   F_j = \frac{\mathit{SS}_R(\beta_j\mid\beta_1,\beta_0)/1}{\mathit{MS}_E(x_j, x_1)},
   $$

   where $\mathit{MS}_E(x_j,x_1)$ is the residual mean square of the model
   containing $x_1$ and $x_j$. Let $j^{*}$ be the candidate with the
   largest $F_j$ (smallest $p$-value). If that $p$-value is below
   $\alpha_{\text{in}}$, add $x_{j^{*}}$.
3. **Remove.** With $x_{j^{*}}$ in the model, check whether $x_1$ is still
   needed:

   $$
   F_1 = \frac{\mathit{SS}_R(\beta_1\mid\beta_{j^{*}},\beta_0)/1}{\mathit{MS}_E(x_1,x_{j^{*}})}.
   $$

   If the $p$-value of $F_1$ exceeds $\alpha_{\text{out}}$, remove $x_1$.
   When several predictors are candidates for removal, compute for each
   predictor $x_j$ in the current model

   $$
   F_j = \frac{\mathit{SS}_R(\beta_j\mid \text{all other coefficients in the model})}{\mathit{MS}_E},
   $$

   and remove the one with the smallest $F_j$ if its $p$-value exceeds
   $\alpha_{\text{out}}$.
4. **Repeat** steps 2 and 3 until no predictor can be added or removed.
   Predictors removed in earlier rounds remain candidates for re-entry.
:::

The degrees of freedom of each $F_j$ depend on the number of predictors in
the current model.

Two simplified versions of this procedure are common:

**Forward selection**
: only adds predictors and never removes them, so it ignores the effect that
  a newly added predictor has on those already in the model.

**Backward elimination**
: starts with all $K$ candidates and removes, one at a time, the predictor
  with the largest $p$-value above $\alpha_{\text{out}}$; it stops when no
  predictor meets the removal criterion.

## Implementation

The R functions `regsubsets()` (package `leaps`) and `step()` that carry
out these procedures are demonstrated in {doc}`25-linear-regression-r`.

## Comments and recommendations

- Check the final model for adequacy: residuals, influential points, and
  overfitting ({doc}`13-prediction-and-model-assessment`).
- Try several approaches to model building and variable selection and check
  whether they lead to consistent answers.
- If the number of candidates is not too large, best subset selection is
  recommended, because it is guaranteed to find the best model of each size
  for the chosen criterion.
- Stepwise methods do not guarantee the "best" model.

## Summary and Common Mistakes
### Summary

- A selection procedure needs a criterion and a search method.
- $R^2_{\mathrm{adj}}$, $\mathit{MS}_E$, $C_p$, AIC, AICc, BIC, and PRESS
  account for model size; $R^2$ does not.
- Best subset selection fits $2^K$ models; forward and backward selection
  fit about $K^2/2$ models.

### Common mistakes

- Using $R^2$ or $\mathit{SS}_E$ to choose between models of different
  sizes.
- Reporting $p$-values of the selected model as if the model had been
  specified in advance; selection makes them overly optimistic.
- Treating the selected subset as the unique true model.

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction* (2nd ed.). Springer. <https://doi.org/10.1007/978-0-387-84858-7>
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An introduction to statistical learning: With applications in R*. Springer. <https://doi.org/10.1007/978-1-4614-7138-7>
- Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley.
- Hurvich, C. M., & Tsai, C.-L. (1989). Regression and time series model selection in small samples. *Biometrika, 76*(2), 297–307. <https://doi.org/10.1093/biomet/76.2.297>
- Soch, J., & The Book of Statistical Proofs contributors. (n.d.). *The book of statistical proofs*. <https://statproofbook.github.io/>
- Wikipedia contributors. (n.d.). Akaike information criterion. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Akaike_information_criterion>
- Wikipedia contributors. (n.d.). Bayesian information criterion. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Bayesian_information_criterion>
- Wikipedia contributors. (n.d.). Mallows's Cp. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Mallows%27s_Cp>

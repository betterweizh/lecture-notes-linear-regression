# 抽样分布与极限定理

## 概述

回归中的检验和置信区间依赖于由正态分布导出的一小类分布：$\chi^2$ 分布、$t$ 分布、$F$ 分布及其非中心形式。本章给出它们的定义，介绍在线性模型中产生这些分布的二次型结论（包括 Cochran 定理），回顾估计与假设检验的基本概念，并总结广义线性模型大样本结论背后的极限定理。 本章以 {doc}`01-probability-and-random-vectors` 以及秩、迹和幂等矩阵（{doc}`05-linear-algebra`）为基础。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 定义 $\chi^2$、$t$、$F$ 分布及其非中心形式；
- 确定正态变量二次型的分布，并运用 Cochran 定理；
- 将估计量的均方误差分解为方差与偏差平方；
- 解释第一类和第二类错误、功效、$p$ 值和置信区间；
- 陈述大数定律、中心极限定理以及极大似然估计量的大样本性质。
:::

## 由正态分布导出的分布

### $\chi^2$ 分布

:::{prf:definition} 卡方分布
:label: def-chi2

若 $Z_1,\ldots,Z_k$ 是相互独立的 $\mathcal{N}(0,1)$ 变量，则

$$
U = \sum_{i=1}^k Z_i^2 \sim \chi^2(k),
$$

称为**自由度为 $k$ 的 $\chi^2$ 分布**（chi-square distribution），且 $\E[U] = k$，$\Var(U) = 2k$。
:::

若 $U_1 \sim \chi^2(k_1)$ 与 $U_2 \sim \chi^2(k_2)$ 独立，则 $U_1 + U_2 \sim \chi^2(k_1 + k_2)$（可加性）。

### $t$ 分布

:::{prf:definition} Student $t$ 分布
:label: def-t

若 $Z \sim \mathcal{N}(0,1)$ 与 $U \sim \chi^2(\nu)$ 独立，则

$$
T = \frac{Z}{\sqrt{U/\nu}} \sim t(\nu),
$$

称为**自由度为 $\nu$ 的 $t$ 分布**。
:::

$t$ 分布关于零对称，尾部比标准正态分布更厚；当 $\nu\to\infty$ 时它收敛于 $\mathcal{N}(0,1)$。

### $F$ 分布

:::{prf:definition} $F$ 分布
:label: def-f

若 $U_1 \sim \chi^2(\nu_1)$ 与 $U_2 \sim \chi^2(\nu_2)$ 独立，则

$$
F = \frac{U_1/\nu_1}{U_2/\nu_2} \sim F(\nu_1, \nu_2),
$$

称为**自由度为 $(\nu_1, \nu_2)$ 的 $F$ 分布**。
:::

若 $T \sim t(\nu)$，则 $T^2 \sim F(1, \nu)$。这就是单个系数的偏 $F$ 检验等价于双侧 $t$ 检验的原因（{doc}`10-anova-and-inference`）。

分位数记为 $\chi^2_{q,\nu}$、$t_{q,\nu}$ 和 $F_{q;\nu_1,\nu_2}$，表示 $q$ 分位数；例如 $t_{1-\alpha/2,\nu}$ 是以概率 $\alpha/2$ 被超过的值。

### 非中心分布

当正态变量的均值不为零时，同样的构造给出非中心分布。它们描述原假设不成立时检验统计量的分布，用于计算功效（{doc}`11-one-way-anova`）。

:::{prf:definition} 非中心 $\chi^2$ 分布
:label: def-noncentral-chi2

若 $X_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$（$i = 1,\ldots,k$）相互独立，则

$$
Y = \sum_{i=1}^k \frac{X_i^2}{\sigma_i^2} \sim \chi'^2(k, \delta),
$$

称为自由度为 $k$ 的**非中心 $\chi^2$ 分布**，其**非中心参数**为

$$
\delta = \sum_{i=1}^k \frac{\mu_i^2}{\sigma_i^2} .
$$

其均值为 $\E[Y] = k + \delta$，故 $\delta = \E[Y] - k$。当 $\delta = 0$ 时退化为中心 $\chi^2(k)$ 分布。
:::

:::{prf:definition} 非中心 $F$ 分布
:label: def-noncentral-f

若 $X_1 \sim \chi'^2(k_1, \delta)$ 与 $X_2 \sim \chi^2(k_2)$ 独立，则

$$
F = \frac{X_1/k_1}{X_2/k_2} \sim F'(k_1, k_2, \delta),
$$

称为自由度为 $(k_1, k_2)$、非中心参数为 $\delta$ 的**非中心 $F$ 分布**。
:::

## 正态向量的二次型

回归中的平方和都是正态向量的二次型 $\bz\T\mathbf{A}\bz$。下列结论给出它们的分布。

:::{prf:theorem} 二次型的分布
:label: thm-quadratic-form-chi2

设 $\bz \sim \mathcal{N}_m(\boldsymbol\mu, \sigma^2\bI_m)$，$\mathbf{A}$ 为秩为 $r$ 的对称幂等矩阵，则

$$
\frac{\bz\T\mathbf{A}\bz}{\sigma^2} \sim \chi'^2\!\left(r,\ \frac{\boldsymbol\mu\T\mathbf{A}\boldsymbol\mu}{\sigma^2}\right),
$$

当 $\mathbf{A}\boldsymbol\mu = \bzero$ 时为中心 $\chi^2(r)$ 分布。
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

由 {prf:ref}`thm-spectral`，秩为 $r$ 的对称幂等矩阵可写成 $\mathbf{A} = \mathbf{Q}_r\mathbf{Q}_r\T$，其中 $\mathbf{Q}_r\in\R^{m\times r}$ 的列标准正交（$\mathbf{Q}_r\T\mathbf{Q}_r = \bI_r$）。令 $\mathbf{w} = \mathbf{Q}_r\T\bz/\sigma$。由 {prf:ref}`thm-mvn-properties`，$\mathbf{w} \sim \mathcal{N}_r(\mathbf{Q}_r\T\boldsymbol\mu/\sigma,\ \bI_r)$，其分量相互独立且方差为 1，并且

$$
\begin{aligned}
\frac{\bz\T\mathbf{A}\bz}{\sigma^2} &= \mathbf{w}\T\mathbf{w} \\
&= \sum_{i=1}^r w_i^2 .
\end{aligned}
$$

由 {prf:ref}`def-noncentral-chi2`，这是自由度为 $r$、非中心参数为 $\lVert\mathbf{Q}_r\T\boldsymbol\mu\rVert^2/\sigma^2 = \boldsymbol\mu\T\mathbf{A}\boldsymbol\mu/\sigma^2$ 的非中心 $\chi^2$ 分布。
:::

:::{prf:theorem} 二次型与线性型的独立性
:label: thm-quadratic-independence

设 $\bz \sim \mathcal{N}_m(\boldsymbol\mu, \sigma^2\bI_m)$，$\mathbf{A}$、$\mathbf{B}$ 为对称矩阵，$\mathbf{L}$ 为矩阵。

1. 若 $\mathbf{A}\mathbf{B} = \mathbf{0}$，则 $\bz\T\mathbf{A}\bz$ 与 $\bz\T\mathbf{B}\bz$ 独立。
2. 若 $\mathbf{L}\mathbf{A} = \mathbf{0}$，则 $\mathbf{L}\bz$ 与 $\bz\T\mathbf{A}\bz$ 独立。
:::

当 $\mathbf{A}$、$\mathbf{B}$ 幂等时，两条结论都由 {prf:ref}`thm-mvn-properties` 的性质 3 得到：每个二次型都是 $\mathbf{A}\bz$（或 $\mathbf{B}\bz$）的函数，而 $\Cov(\mathbf{A}\bz, \mathbf{B}\bz) = \sigma^2\mathbf{A}\mathbf{B}$。

:::{prf:theorem} Cochran 定理
:label: thm-cochran

设 $\bz \sim \mathcal{N}_m(\bzero, \bI_m)$，且

$$
\bz\T\bz = \sum_{l=1}^{L} \bz\T\mathbf{A}_l\bz,
$$

其中每个 $\mathbf{A}_l$ 对称、秩为 $r_l$。若 $r_1 + \cdots + r_L = m$，则各二次型 $\bz\T\mathbf{A}_l\bz$ 相互独立，且对每个 $l$ 有 $\bz\T\mathbf{A}_l\bz \sim \chi^2(r_l)$。
:::

Cochran 定理正是方差分析表中各平方和为相互独立的 $\chi^2$ 变量、且自由度可以相加的理论依据。

:::{prf:example} 样本均值与样本方差
:label: ex-sample-variance

设 $Y_1,\ldots,Y_n$ 独立同分布于 $\mathcal{N}(\mu,\sigma^2)$，$Z_i = (Y_i - \mu)/\sigma$。记 $\mathbf{J}_n = \mathbf{1}_n\mathbf{1}_n\T$，则

$$
\bz\T\bz = \bz\T\Bigl(\bI_n - \tfrac1n\mathbf{J}_n\Bigr)\bz + \bz\T\Bigl(\tfrac1n\mathbf{J}_n\Bigr)\bz ,
$$

两个矩阵的秩分别为 $n-1$ 和 $1$。由 Cochran 定理，

$$
\begin{gathered}
\frac{(n-1)S^2}{\sigma^2}
= \frac{1}{\sigma^2}\sum_{i=1}^n (Y_i - \bar{Y})^2 \sim \chi^2(n-1), \\
\frac{n(\bar{Y}-\mu)^2}{\sigma^2} \sim \chi^2(1),
\end{gathered}
$$

且二者独立，因此样本均值 $\bar{Y}$ 与样本方差 $S^2$ 相互独立。
:::

## 估计

参数 $\theta$ 的**估计量** $\widehat{\theta}$ 是数据的函数；它对某一组数据的取值称为**估计值**。

- **偏差**（bias）为 $\operatorname{Bias}(\widehat\theta) = \E[\widehat\theta] - \theta$；偏差为零时称 $\widehat\theta$ **无偏**。
- **标准误** $\mathrm{SE}(\widehat\theta)$ 是 $\widehat\theta$ 标准差的估计。
- 若 $n \to \infty$ 时 $\widehat\theta_n$ 依概率收敛于 $\theta$，则称其**相合**（consistent）。

:::{prf:proposition} 均方误差的偏差–方差分解
:label: prop-mse-decomposition

估计量的**均方误差**（mean squared error, MSE）满足

$$
\begin{aligned}
\operatorname{MSE}(\widehat\theta)
&= \E\bigl[(\widehat\theta - \theta)^2\bigr] \\
&= \Var(\widehat\theta) + \bigl[\operatorname{Bias}(\widehat\theta)\bigr]^2 .
\end{aligned}
$$

对向量估计量 $\widehat{\boldsymbol\theta}$，

$$
\begin{aligned}
\E\bigl[\lVert\widehat{\boldsymbol\theta} - \boldsymbol\theta\rVert_2^2\bigr]
&= \tr\bigl(\Cov(\widehat{\boldsymbol\theta})\bigr)
 + \bigl\lVert\E[\widehat{\boldsymbol\theta}] - \boldsymbol\theta\bigr\rVert_2^2 .
\end{aligned}
$$
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

加减 $\E[\widehat\theta]$：

$$
\begin{aligned}
\E\bigl[(\widehat\theta - \theta)^2\bigr]
&= \E\bigl[(\widehat\theta - \E[\widehat\theta])^2\bigr]
 + 2\,\bigl(\E[\widehat\theta] - \theta\bigr)\,\E\bigl[\widehat\theta - \E[\widehat\theta]\bigr]
 + \bigl(\E[\widehat\theta] - \theta\bigr)^2 \\
&= \Var(\widehat\theta) + 0 + \bigl[\operatorname{Bias}(\widehat\theta)\bigr]^2 .
\end{aligned}
$$

对向量情形，将上式用于每个分量后求和即可。
:::

无偏估计量不一定最好：方差小得多的有偏估计量可能具有更小的 MSE。岭回归和 lasso 回归正是利用了这种权衡（{doc}`16-ridge-and-lasso`）。

### 极大似然

对密度为 $f(y_i;\boldsymbol\theta)$ 的独立观测，**似然函数**为 $\mathcal{L}(\boldsymbol\theta) = \prod_i f(y_i;\boldsymbol\theta)$，**对数似然函数**为 $\ell(\boldsymbol\theta) = \ln\mathcal{L}(\boldsymbol\theta)$。**极大似然估计量**（MLE）使 $\ell$ 最大。**得分**为 $\nabla\ell(\boldsymbol\theta)$，**Fisher 信息**为

$$
\mathbf{I}(\boldsymbol\theta) = -\E\left[\frac{\partial^2\ell(\boldsymbol\theta)}{\partial\boldsymbol\theta\,\partial\boldsymbol\theta\T}\right].
$$

:::{prf:theorem} 极大似然估计量的大样本性质
:label: thm-mle-asymptotic

在正则条件下，当 $n \to \infty$ 时：

1. MLE 是相合的；
2. MLE 渐近正态，$\widehat{\boldsymbol\theta} \overset{\text{近似}}{\sim} \mathcal{N}\bigl(\boldsymbol\theta,\ \mathbf{I}(\boldsymbol\theta)^{-1}\bigr)$，这为 Wald 检验和 Wald 区间提供了依据；
3. **（Wilks）** 对嵌套模型，似然比统计量 $-2\bigl[\ell(\widehat{\boldsymbol\theta}_0) - \ell(\widehat{\boldsymbol\theta})\bigr]$ 在原假设下近似服从 $\chi^2(d)$，其中 $\widehat{\boldsymbol\theta}_0$ 为原假设下的 MLE，$d$ 为约束个数。
:::

这些结论是 {doc}`17-generalized-linear-models` 及后续各章中 Wald 检验和偏差下降检验的基础。

## 假设检验与置信区间

对**原假设** $H_0$ 与**备择假设** $H_1$ 的检验使用检验统计量和拒绝域。

| | $H_0$ 为真 | $H_0$ 为假 |
|---|---|---|
| 拒绝 $H_0$ | **第一类错误**（概率 $\alpha$） | 正确决策（概率 $1-\beta$，即**功效**） |
| 不拒绝 $H_0$ | 正确决策 | **第二类错误**（概率 $\beta$） |

- **显著性水平** $\alpha$ 是我们能接受的第一类错误的最大概率。
- **$p$ 值**是在 $H_0$ 下计算的、检验统计量至少与观测值一样极端的概率。当 $p < \alpha$ 时在水平 $\alpha$ 下拒绝 $H_0$。
- **$100(1-\alpha)\%$ 置信区间**是一个随机区间，在重复抽样中以概率 $1-\alpha$ 包含真实参数。它由双侧水平 $\alpha$ 检验不会拒绝的参数值组成。

## 极限定理

:::{prf:theorem} 大数定律
:label: thm-lln

若 $X_1, X_2,\ldots$ 独立同分布且均值 $\mu$ 有限，则当 $n \to \infty$ 时，$\bar{X}_n = \frac1n\sum_{i=1}^n X_i$ 依概率（实际上几乎必然）收敛于 $\mu$。
:::

:::{prf:theorem} 中心极限定理
:label: thm-clt

若 $X_1, X_2,\ldots$ 独立同分布，均值为 $\mu$，方差 $\sigma^2 > 0$ 有限，则

$$
\frac{\sqrt{n}\,(\bar{X}_n - \mu)}{\sigma} \xrightarrow{d} \mathcal{N}(0,1)
\qquad (n \to \infty).
$$
:::

**Slutsky 定理**允许用相合估计代替未知量：若 $U_n \xrightarrow{d} U$ 且 $V_n$ 依概率收敛于 $c \ne 0$，则 $U_n/V_n \xrightarrow{d} U/c$。结合中心极限定理，它解释了为什么即使误差不服从正态分布，大样本下 $t$ 统计量也近似服从正态分布。

## 本章回顾
### 小结

- $\chi^2(k)$ 是 $k$ 个独立标准正态变量的平方和；$t(\nu) = Z/\sqrt{U/\nu}$；$F(\nu_1,\nu_2)$ 是两个独立 $\chi^2$ 变量各自除以自由度之比。均值非零时得到非中心形式。
- 若 $\mathbf{A}$ 对称幂等、秩为 $r$，则二次型 $\bz\T\mathbf{A}\bz/\sigma^2$ 服从（非中心）$\chi^2(r)$；Cochran 定理给出方差分析平方和的独立性。
- $\operatorname{MSE} = \Var + \operatorname{Bias}^2$。
- MLE 相合且渐近正态，似然比统计量渐近服从 $\chi^2$ 分布。

### 常见错误

- 未检查分子与分母是否独立就使用 $t$ 或 $F$ 分布。
- 把 $p$ 值理解为 $H_0$ 为真的概率。
- 认为无偏估计量的 MSE 总是最小。

## 参考文献

- Casella, G., & Berger, R. L. (2002). *Statistical inference* (2nd ed.). Duxbury.
- Seber, G. A. F., & Lee, A. J. (2003). *Linear regression analysis* (2nd ed.). Wiley. <https://doi.org/10.1002/9780471722199>
- Wikipedia contributors. (n.d.). Noncentral chi-squared distribution. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution>
- Wikipedia contributors. (n.d.). Noncentral F-distribution. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Noncentral_F-distribution>

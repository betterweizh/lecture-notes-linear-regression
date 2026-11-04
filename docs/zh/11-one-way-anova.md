# 单因素方差分析

## 概述

**单因素方差分析**（one-way analysis of variance, ANOVA）比较 $a$ 个组的均值，例如设计试验中 $a$ 种处理下的产量。它是线性模型的一个特例，其中预测变量是组别示性变量，其 $F$ 检验就是 {doc}`10-anova-and-inference` 中的整体 $F$ 检验。本章给出固定效应模型，证明平方和分解和各平方和的分布，并推导期望均方；然后给出处理均值的估计及其置信区间，用 Fisher 最小显著差异法比较均值对，把公式推广到非平衡设计，用残差检查模型假设，并利用非中心 $F$ 分布计算功效、确定样本量。随机效应模型和区组设计见 {doc}`12-random-effects-and-blocking`。 本章以非中心分布和 Cochran 定理（{doc}`02-sampling-distributions-and-limit-theorems`）以及 {doc}`10-anova-and-inference` 为基础。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 写出单因素固定效应模型，并把它表示为对示性变量的线性回归；
- 证明 $\mathit{SS}_T = \mathit{SS}_{\mathrm{Treatments}} + \mathit{SS}_E$；
- 推导 $\mathit{SS}_{\mathrm{Treatments}}$ 和 $\mathit{SS}_E$ 的分布以及期望均方；
- 进行处理均值相等的 $F$ 检验；
- 用置信区间估计处理均值及其差，并用 Fisher LSD 法比较均值对；
- 用残差图检查模型假设；
- 用非中心 $F$ 分布计算检验的第二类错误概率和功效，并据此确定重复次数。
:::

## 符号说明

| 符号 | 含义 |
|---|---|
| $a$ | 处理（组）的个数 |
| $n$ | 每种处理下的观测数（平衡设计） |
| $N = an$ | 总观测数 |
| $y_{ij}$ | 第 $i$ 种处理下的第 $j$ 个观测，$i = 1,\ldots,a$，$j = 1,\ldots,n$ |
| $\mu$, $\tau_i$ | 总均值与第 $i$ 种处理的效应 |
| $y_{i\cdot} = \sum_j y_{ij}$, $\bar{y}_{i\cdot} = y_{i\cdot}/n$ | 第 $i$ 种处理的总和与均值 |
| $y_{\cdot\cdot} = \sum_i\sum_j y_{ij}$, $\bar{y}_{\cdot\cdot} = y_{\cdot\cdot}/N$ | 总和与总均值 |

下标中的圆点表示对该下标求和。

## 单因素试验

在**单因素试验**（single-factor experiment）中，一个因素取 $a$ 个水平，称为**处理**（treatment）；每种处理施加于 $n$ 个试验单元，同一处理下的观测称为该处理的**重复**（replicate）。各次试验按**随机顺序**进行，使不受控制的影响（设备漂移、操作人员或原材料的变化）平均地分散到各处理中，而不与处理混杂。一项试验依次经历推测（引发试验的假设）、试验本身、统计分析和结论；结论往往又引出修正的推测和新的试验。

数据按每种处理一行排列：

| 处理 | 观测 | 总和 | 均值 |
|---|---|---|---|
| $1$ | $y_{11},\ y_{12},\ \ldots,\ y_{1n}$ | $y_{1\cdot}$ | $\bar{y}_{1\cdot}$ |
| $2$ | $y_{21},\ y_{22},\ \ldots,\ y_{2n}$ | $y_{2\cdot}$ | $\bar{y}_{2\cdot}$ |
| $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ |
| $a$ | $y_{a1},\ y_{a2},\ \ldots,\ y_{an}$ | $y_{a\cdot}$ | $\bar{y}_{a\cdot}$ |
| | | $y_{\cdot\cdot}$ | $\bar{y}_{\cdot\cdot}$ |

若这 $a$ 个水平是试验者特意选定的，则处理是**固定**的，结论只适用于这些水平，模型参数 $\mu$、$\tau_i$ 和 $\sigma^2$ 本身就是关注对象；若这 $a$ 个水平是从更大的水平总体中随机抽取的，则处理是**随机**的，关注的是整个总体的变异（{doc}`12-random-effects-and-blocking`）。本章讨论固定情形。

## 固定效应模型

:::{prf:assumption} 单因素固定效应模型
:label: asm-one-way

$$
y_{ij} = \mu + \tau_i + \varepsilon_{ij},
\qquad i = 1,\ldots,a,\quad j = 1,\ldots,n,
$$

其中处理效应（treatment effect）是满足 $\sum_{i=1}^a\tau_i = 0$ 的固定常数，误差 $\varepsilon_{ij}$ 独立同分布于 $\mathcal{N}(0,\sigma^2)$。
:::

**处理均值**为 $\mu_i = \mu + \tau_i$。约束 $\sum_i\tau_i = 0$ 使 $\mu$ 成为各处理均值的平均值，并使参数可识别。我们关心的问题是各处理是否有差异：

$$
\begin{gathered}
H_0 :\ \tau_1 = \tau_2 = \cdots = \tau_a = 0 \quad\text{对} \\
H_1 :\ \tau_i \ne 0 \text{ 至少对一个 } i \text{ 成立},
\end{gathered}
$$

等价地，$H_0:\ \mu_1 = \mu_2 = \cdots = \mu_a$。

### 作为线性回归的模型

定义 $a-1$ 个示性（虚拟）变量：若观测属于第 $l$ 种处理，则 $x_{(l)} = 1$，否则为 $0$，$l = 1,\ldots,a-1$，以第 $a$ 种处理为参照类别。于是

$$
y = \beta_0 + \beta_1x_{(1)} + \cdots + \beta_{a-1}x_{(a-1)} + \varepsilon,
$$

其中 $\beta_0 = \mu_a$，$\beta_l = \mu_l - \mu_a$。所有处理均值相等恰好等价于 $\beta_1 = \cdots = \beta_{a-1} = 0$，因此下文的单因素方差分析 $F$ 检验就是这个回归（$p = a$ 个系数）的整体 $F$ 检验 {eq}`eq-overall-f`。

## 平方和分解

定义

$$
\begin{gathered}
\mathit{SS}_T = \sum_{i=1}^a\sum_{j=1}^n \bigl(y_{ij} - \bar{y}_{\cdot\cdot}\bigr)^2, \\
\mathit{SS}_{\mathrm{Treatments}} = n\sum_{i=1}^a \bigl(\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot}\bigr)^2, \\
\mathit{SS}_E = \sum_{i=1}^a\sum_{j=1}^n \bigl(y_{ij} - \bar{y}_{i\cdot}\bigr)^2 .
\end{gathered}
$$

$\mathit{SS}_{\mathrm{Treatments}}$ 衡量处理**之间**的变异，$\mathit{SS}_E$ 衡量处理**内部**的变异。

:::{prf:theorem} 单因素方差分析恒等式
:label: thm-one-way-decomposition

$$
\mathit{SS}_T = \mathit{SS}_{\mathrm{Treatments}} + \mathit{SS}_E .
$$
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

将 $y_{ij} - \bar{y}_{\cdot\cdot}$ 写成 $(\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot}) + (y_{ij} - \bar{y}_{i\cdot})$ 并展开平方：

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

这是因为对每个 $i$ 都有 $\sum_{j=1}^n (y_{ij} - \bar{y}_{i\cdot}) = 0$。
:::

各平方和的自由度分别为 $N-1$、$a-1$ 和 $N-a$，它们同样满足相加关系。均方为 $\mathit{MS}_{\mathrm{Treatments}} = \mathit{SS}_{\mathrm{Treatments}}/(a-1)$ 和 $\mathit{MS}_E = \mathit{SS}_E/(N-a)$。

## 期望均方

:::{prf:theorem} 期望均方
:label: thm-one-way-ems

在 {prf:ref}`asm-one-way` 下，

$$
\begin{gathered}
\E[\mathit{MS}_E] = \sigma^2, \\
\E[\mathit{MS}_{\mathrm{Treatments}}] = \sigma^2 + \frac{n}{a-1}\sum_{i=1}^a\tau_i^2 .
\end{gathered}
$$
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

**误差均方。** 展开平方，并利用 $\sum_j y_{ij}\bar{y}_{i\cdot} = n\bar{y}_{i\cdot}^2 = y_{i\cdot}^2/n$，

$$
\begin{aligned}
\mathit{SS}_E
&= \sum_{i=1}^a\sum_{j=1}^n \bigl(y_{ij}^2 - 2y_{ij}\bar{y}_{i\cdot} + \bar{y}_{i\cdot}^2\bigr) \\
&= \sum_{i=1}^a\sum_{j=1}^n y_{ij}^2 - 2n\sum_{i=1}^a\bar{y}_{i\cdot}^2 + n\sum_{i=1}^a\bar{y}_{i\cdot}^2 \\
&= \sum_{i=1}^a\sum_{j=1}^n y_{ij}^2 - \frac1n\sum_{i=1}^a y_{i\cdot}^2 .
\end{aligned}
$$

代入 $y_{ij} = \mu + \tau_i + \varepsilon_{ij}$，并利用 $\E[\varepsilon_{ij}] = 0$、$\E[\varepsilon_{ij}^2] = \sigma^2$ 以及误差的独立性，

$$
\begin{aligned}
\E\Bigl[\sum_{i=1}^a\sum_{j=1}^n y_{ij}^2\Bigr] &= n\sum_{i=1}^a(\mu+\tau_i)^2 + N\sigma^2, \\
\E\Bigl[\frac1n\sum_{i=1}^a y_{i\cdot}^2\Bigr]
&= \frac1n\sum_{i=1}^a \E\Bigl[\Bigl(n(\mu+\tau_i) + \sum_{j=1}^n\varepsilon_{ij}\Bigr)^2\Bigr] \\
&= n\sum_{i=1}^a(\mu+\tau_i)^2 + a\sigma^2 .
\end{aligned}
$$

因此 $\E[\mathit{SS}_E] = (N-a)\sigma^2$，$\E[\mathit{MS}_E] = \sigma^2$。

**处理均方。** 由于 $\sum_i\bar{y}_{i\cdot}\bar{y}_{\cdot\cdot} = a\bar{y}_{\cdot\cdot}^2$，

$$
\begin{aligned}
\mathit{MS}_{\mathrm{Treatments}}
&= \frac{n}{a-1}\sum_{i=1}^a \bigl(\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot}\bigr)^2 \\
&= \frac{n}{a-1}\left[\sum_{i=1}^a\bar{y}_{i\cdot}^2 - a\,\bar{y}_{\cdot\cdot}^2\right].
\end{aligned}
$$

每个处理均值满足 $\bar{y}_{i\cdot} \sim \mathcal{N}(\mu+\tau_i,\ \sigma^2/n)$；又因为 $\sum_i\tau_i = 0$，总均值满足 $\bar{y}_{\cdot\cdot} \sim \mathcal{N}(\mu,\ \sigma^2/N)$。利用 $\E[W^2] = (\E[W])^2 + \Var(W)$，

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

无论 $H_0$ 是否成立，$\mathit{MS}_E$ 都是 $\sigma^2$ 的估计；而 $\mathit{MS}_{\mathrm{Treatments}}$ 只在 $H_0$ 下估计 $\sigma^2$，否则会因处理间的差异而偏大。因此二者之比是自然的检验统计量。

## 平方和的分布

:::{prf:theorem} 单因素方差分析中的分布
:label: thm-one-way-distributions

在 {prf:ref}`asm-one-way` 下：

1. $\mathit{SS}_E/\sigma^2 \sim \chi^2(N-a)$；
2. $\mathit{SS}_{\mathrm{Treatments}}$ 与 $\mathit{SS}_E$ 相互独立；
3. 若 $H_0$ 成立，则 $\mathit{SS}_{\mathrm{Treatments}}/\sigma^2 \sim \chi^2(a-1)$。
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

**第 1 条。** 在第 $i$ 种处理内，观测 $y_{i1},\ldots,y_{in}$ 独立同分布于 $\mathcal{N}(\mu+\tau_i, \sigma^2)$。由 {prf:ref}`ex-sample-variance`，

$$
\frac{1}{\sigma^2}\sum_{j=1}^n (y_{ij} - \bar{y}_{i\cdot})^2 \sim \chi^2(n-1),
$$

且该组内平方和与 $\bar{y}_{i\cdot}$ 独立。$a$ 个组相互独立，由独立 $\chi^2$ 变量的可加性，

$$
\begin{aligned}
\frac{\mathit{SS}_E}{\sigma^2} &= \sum_{i=1}^a \frac{1}{\sigma^2}\sum_{j=1}^n (y_{ij} - \bar{y}_{i\cdot})^2 \\
&\sim \chi^2\bigl(a(n-1)\bigr) = \chi^2(N-a).
\end{aligned}
$$

**第 2 条。** $\mathit{SS}_{\mathrm{Treatments}}$ 是处理均值 $\bar{y}_{1\cdot},\ldots,\bar{y}_{a\cdot}$ 的函数，$\mathit{SS}_E$ 是各组内平方和的函数。由第 1 条，每个组内平方和与本组均值独立，而各组之间相互独立，所以这两组量相互独立，从而 $\mathit{SS}_{\mathrm{Treatments}}$ 与 $\mathit{SS}_E$ 独立。

**第 3 条。** 在 $H_0$ 下，$y_{ij} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(\mu,\sigma^2)$，故各处理均值独立同分布于 $\mathcal{N}(\mu, \sigma^2/n)$。令

$$
z_i = \frac{\bar{y}_{i\cdot} - \mu}{\sqrt{\sigma^2/n}} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,1),
\qquad i = 1,\ldots,a,
$$

并记 $\bar{z} = \frac1a\sum_i z_i$。由于 $\bar{y}_{\cdot\cdot}$ 是各处理均值的平均，$z_i - \bar{z} = (\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot})/\sqrt{\sigma^2/n}$，于是

$$
\begin{aligned}
\sum_{i=1}^a z_i^2
&= \sum_{i=1}^a (z_i - \bar{z})^2 + a\,\bar{z}^2 \\
&= \frac{\mathit{SS}_{\mathrm{Treatments}}}{\sigma^2} + \frac{1}{a}\Bigl(\sum_{i=1}^a z_i\Bigr)^2 \\
&= \bz\T\Bigl(\bI_a - \tfrac1a\mathbf{J}_a\Bigr)\bz + \bz\T\Bigl(\tfrac1a\mathbf{J}_a\Bigr)\bz ,
\end{aligned}
$$

其中 $\bz = (z_1,\ldots,z_a)\T$，$\mathbf{J}_a$ 为 $a\times a$ 全 1 矩阵。两个矩阵的秩分别为 $a-1$ 和 $1$，之和为 $a$，由 Cochran 定理（{prf:ref}`thm-cochran`），

$$
\frac{\mathit{SS}_{\mathrm{Treatments}}}{\sigma^2} = \sum_{i=1}^a (z_i - \bar{z})^2 \sim \chi^2(a-1).
$$
:::

## $F$ 检验

由 {prf:ref}`thm-one-way-distributions` 和 {prf:ref}`def-f`，在 $H_0$ 下

$$
\begin{aligned}
F_0 &= \frac{\mathit{MS}_{\mathrm{Treatments}}}{\mathit{MS}_E} \\
    &= \frac{\bigl(\mathit{SS}_{\mathrm{Treatments}}/\sigma^2\bigr)/(a-1)}{\bigl(\mathit{SS}_E/\sigma^2\bigr)/(N-a)}
    \ \sim\ F(a-1,\ N-a).
\end{aligned}
$$ (eq-one-way-f)

当 $F_0 > F_{1-\alpha;\,a-1,\,N-a}$ 时，在水平 $\alpha$ 下拒绝 $H_0$。结果汇总于方差分析表。

| 变异来源 | 平方和 | 自由度 | 均方 | $F_0$ |
|---|---|---|---|---|
| 处理 | $\mathit{SS}_{\mathrm{Treatments}}$ | $a-1$ | $\mathit{MS}_{\mathrm{Treatments}}$ | $\mathit{MS}_{\mathrm{Treatments}}/\mathit{MS}_E$ |
| 误差 | $\mathit{SS}_E$ | $N-a$ | $\mathit{MS}_E$ | |
| 总计 | $\mathit{SS}_T$ | $N-1$ | | |

## 估计模型参数

总均值、处理效应和处理均值的最小二乘估计为

$$
\begin{aligned}
\widehat{\mu} &= \bar{y}_{\cdot\cdot}, \\
\widehat{\tau}_i &= \bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot}, \\
\widehat{\mu}_i &= \widehat{\mu} + \widehat{\tau}_i \\
&= \bar{y}_{i\cdot},
\end{aligned}
$$

$\sigma^2$ 的估计为 $\widehat{\sigma}^2 = \mathit{MS}_E$（{prf:ref}`thm-one-way-ems`）。这些估计满足 $\sum_i\widehat{\tau}_i = 0$，与 {prf:ref}`asm-one-way` 中的约束一致。

### 处理均值的置信区间

在 {prf:ref}`asm-one-way` 下，$\bar{y}_{i\cdot} \sim \mathcal{N}(\mu_i, \sigma^2/n)$，它与 $\mathit{MS}_E$ 独立（{prf:ref}`thm-one-way-distributions` 第 2 条），且 $(N-a)\mathit{MS}_E/\sigma^2 \sim \chi^2(N-a)$，于是由 {prf:ref}`def-t`，

$$
T = \frac{\bar{y}_{i\cdot} - \mu_i}{\sqrt{\mathit{MS}_E/n}} \sim t(N-a).
$$

因此 $\mu_i$ 的 $100(1-\alpha)\%$ 置信区间为

$$
\bar{y}_{i\cdot} \pm t_{1-\alpha/2,\,N-a}\sqrt{\frac{\mathit{MS}_E}{n}} .
$$ (eq-one-way-ci-mean)

### 处理均值之差的置信区间

对 $i \ne j$，$\bar{y}_{i\cdot} - \bar{y}_{j\cdot} \sim \mathcal{N}(\mu_i - \mu_j,\ 2\sigma^2/n)$，同样的论证给出 $100(1-\alpha)\%$ 置信区间

$$
\bigl(\bar{y}_{i\cdot} - \bar{y}_{j\cdot}\bigr) \pm t_{1-\alpha/2,\,N-a}\sqrt{\frac{2\,\mathit{MS}_E}{n}} .
$$ (eq-one-way-ci-diff)

## 多重比较

$F$ 检验拒绝 $H_0:\mu_1 = \cdots = \mu_a$ 后，并不能说明哪些均值不同。**多重比较方法**（multiple comparisons）回答这个问题，其中最简单的是 **Fisher 最小显著差异法**（least significant difference, LSD），它对每一对均值检验

$$
H_0:\ \mu_i = \mu_j \quad\text{对}\quad H_1:\ \mu_i \ne \mu_j
$$

所用统计量为

$$
t_0 = \frac{\bar{y}_{i\cdot} - \bar{y}_{j\cdot}}{\sqrt{2\,\mathit{MS}_E/n}} ,
$$

在 $H_0$ 下它服从 $t(N-a)$。等价地，当 $\lvert\bar{y}_{i\cdot} - \bar{y}_{j\cdot}\rvert$ 超过**最小显著差异**

$$
\mathrm{LSD} = t_{1-\alpha/2,\,N-a}\sqrt{\frac{2\,\mathit{MS}_E}{n}} .
$$ (eq-lsd)

时，判定这对均值有显著差异。这等价于检查区间 {eq}`eq-one-way-ci-diff` 是否不包含零。由于要检验许多对，每对的水平都是 $\alpha$，至少出现一次错误拒绝的概率会超过 $\alpha$；Tukey 法等方法可以控制这一总体错误率。

## 非平衡设计

若处理 $i$ 有 $n_i$ 个观测，$N = \sum_i n_i$，则在各求和式中把 $n$ 换成 $n_i$ 后，分解式 {prf:ref}`thm-one-way-decomposition` 仍然成立。方便的计算公式为

$$
\begin{gathered}
\mathit{SS}_T = \sum_{i=1}^a\sum_{j=1}^{n_i} y_{ij}^2 - \frac{y_{\cdot\cdot}^2}{N}, \\
\mathit{SS}_{\mathrm{Treatments}} = \sum_{i=1}^a\frac{y_{i\cdot}^2}{n_i} - \frac{y_{\cdot\cdot}^2}{N}, \\
\mathit{SS}_E = \mathit{SS}_T - \mathit{SS}_{\mathrm{Treatments}},
\end{gathered}
$$

自由度分别为 $N-1$、$a-1$ 和 $N-a$；$F$ 检验 {eq}`eq-one-way-f` 不变。在区间和 LSD 中，把 $2/n$ 换成 $1/n_i + 1/n_j$：

$$
\mathrm{LSD}_{ij} = t_{1-\alpha/2,\,N-a}\sqrt{\mathit{MS}_E\left(\frac{1}{n_i} + \frac{1}{n_j}\right)} .
$$

在可能的情况下应采用平衡设计：此时检验对方差不等不太敏感，且在 $N$ 给定时功效最大。

## 检查模型的适当性

上述结论依赖于误差独立、服从正态分布且方差相同（{prf:ref}`asm-one-way`）。这些假设用**残差**来检查：

$$
\begin{aligned}
e_{ij} &= y_{ij} - \widehat{y}_{ij}, \\
\widehat{y}_{ij} &= \widehat{\mu} + \widehat{\tau}_i \\
&= \bar{y}_{i\cdot} .
\end{aligned}
$$

若模型适当，残差应当没有结构，即不呈现明显的模式。常用的检查包括：

- **正态性。** 残差的正态概率图（正态 Q–Q 图）应大致为一条直线（{doc}`04-goodness-of-fit-tests`）。适度偏离正态对固定效应 $F$ 检验影响不大。
- **异常值。** 标准化残差 $d_{ij} = e_{ij}/\sqrt{\mathit{MS}_E}$ 近似服从 $\mathcal{N}(0,1)$，因此约 68% 应落在 $[-1,1]$ 内、约 95% 落在 $[-2,2]$ 内；$\lvert d_{ij}\rvert > 3$ 的值可能是异常值，应加以调查。
- **独立性。** 按数据收集的时间顺序画残差图可以发现相关性：连续出现正残差或负残差，说明误差可能不独立。
- **方差齐性。** 残差对拟合值 $\widehat{y}_{ij}$ 的散点图应没有模式；呈漏斗形说明方差随均值变化。

假设不成立时如何处理（变换、稳健或非参数检验）超出了本章的范围。

## 第二类错误、功效与样本量

当 $H_0$ 不成立时，$\mathit{SS}_{\mathrm{Treatments}}/\sigma^2$ 服从非中心 $\chi^2$ 分布，$F_0$ 服从非中心 $F$ 分布（{prf:ref}`def-noncentral-f`）：

$$
\begin{gathered}
F_0 \sim F'\!\left(a-1,\ N-a,\ \delta\right), \\
\delta = \frac{n}{\sigma^2}\sum_{i=1}^a\tau_i^2 .
\end{gathered}
$$

当 $\delta = 0$ 时即为 {eq}`eq-one-way-f` 中的中心 $F$ 分布。**第二类错误**（未能拒绝错误的 $H_0$）的概率为

$$
\begin{aligned}
\beta &= 1 - \Prob\bigl(F_0 > F_{1-\alpha;\,a-1,\,N-a} \mid H_0 \text{ 不成立}\bigr) \\
&= \Prob\bigl(F_0 \le F_{1-\alpha;\,a-1,\,N-a} \mid H_0 \text{ 不成立}\bigr),
\end{aligned}
$$

检验的**功效**为 $1-\beta$。它随重复次数 $n$ 的增大以及处理效应相对于 $\sigma^2$ 的增大而增大。工作特性（OC）曲线把 $\beta$ 画成如下参数的函数：

$$
\begin{aligned}
\Phi^2 &= \frac{n\sum_{i=1}^a\tau_i^2}{a\,\sigma^2} \\
&= \frac{\delta}{a} .
\end{aligned}
$$

### 确定重复次数

试验的**灵敏度**是试验者希望检测到的最小均值差。若任意两个处理均值至少相差 $D$，则 $\sum_i\tau_i^2 \ge D^2/2$，因此 $\Phi^2$ 的最小值为

$$
\Phi^2 = \frac{n\,D^2}{2a\,\sigma^2} .
$$

实际中，先确定 $\alpha$、比值 $D/\sigma$ 和所需功效，再逐步增大 $n$，直到由非中心 $F$ 分布计算（或从 OC 曲线读出）的功效足够大。大多数统计软件都能完成这一计算。

## 本章回顾
### 小结

- 单因素固定效应模型 $y_{ij} = \mu + \tau_i + \varepsilon_{ij}$ 是对 $a-1$ 个示性变量的线性回归。
- $\mathit{SS}_T = \mathit{SS}_{\mathrm{Treatments}} + \mathit{SS}_E$，自由度满足 $N-1 = (a-1) + (N-a)$。
- $\E[\mathit{MS}_E] = \sigma^2$，$\E[\mathit{MS}_{\mathrm{Treatments}}] = \sigma^2 + n\sum_i\tau_i^2/(a-1)$。
- $F_0 = \mathit{MS}_{\mathrm{Treatments}}/\mathit{MS}_E$ 在 $H_0$ 下服从 $F(a-1,N-a)$，否则服从 $F'(a-1,N-a,\delta)$，由此可计算功效和所需的重复次数。
- $\widehat{\mu}_i = \bar{y}_{i\cdot}$，其区间为 $\bar{y}_{i\cdot} \pm t_{1-\alpha/2,N-a}\sqrt{\mathit{MS}_E/n}$；均值对用 LSD 比较。
- 残差图用于检查正态性、异常值、独立性和方差齐性。

### 常见错误

- 忘记约束 $\sum_i\tau_i = 0$；没有它，参数 $\mu,\tau_1,\ldots,\tau_a$ 不可识别。
- 由显著的 $F$ 检验直接得出哪些处理之间有差异；这需要多重比较方法。
- 未把 $n$ 换成各组样本量 $n_i$，就把平衡设计的公式用于非平衡设计。
- 进行全部成对 LSD 比较时，没有注意到总体错误拒绝的概率会超过 $\alpha$。
- 把功效写成 $1-\alpha$；功效是 $1-\beta$。

## 参考文献

- Montgomery, D. C. (2017). *Design and analysis of experiments* (9th ed.). Wiley.
- Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley.
- Wikipedia contributors. (n.d.). Noncentral chi-squared distribution. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution>
- Wikipedia contributors. (n.d.). Noncentral F-distribution. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Noncentral_F-distribution>

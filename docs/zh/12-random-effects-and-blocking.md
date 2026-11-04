# 随机效应与区组设计

## 概述

{doc}`11-one-way-anova` 把 $a$ 种处理看作试验者选定的固定水平。本章讨论单因素试验的两种推广。在**随机效应模型**（random-effects model）中，处理是从更大的水平总体中随机抽取的样本，关心的是响应的变异中有多少来自该因素。在**随机化完全区组设计**（randomized complete block design, RCBD）中，通过在每个区组内试验全部处理来控制干扰因素，否则它会增大误差。两者使用与单因素方差分析相同的平方和，但假设和期望均方不同。 本章以平方和、$F$ 检验和 LSD（{doc}`11-one-way-anova`）以及独立随机变量之和的方差（{doc}`01-probability-and-random-vectors`）为基础。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 区分固定因素与随机因素，写出随机效应模型及其方差分量；
- 推导随机效应模型的期望均方，并检验 $H_0:\sigma_\tau^2 = 0$；
- 估计方差分量，包括各组样本量不等的情形；
- 说明何时以及为何要对干扰因素划分区组；
- 用方差分析表分析随机化完全区组设计，并用 Fisher LSD 比较处理均值。
:::

## 符号说明

| 符号 | 含义 |
|---|---|
| $a$, $n$, $N = an$ | 处理数、每种处理的重复数、观测总数 |
| $\tau_i$ | 处理 $i$ 的效应（在随机效应模型中为随机变量） |
| $\sigma^2$, $\sigma_\tau^2$ | 误差方差与处理方差分量 |
| $b$ | 随机化完全区组设计中的区组数 |
| $\beta_j$ | 区组 $j$ 的效应 |
| $\bar{y}_{i\cdot}$, $\bar{y}_{\cdot j}$, $\bar{y}_{\cdot\cdot}$ | 处理均值、区组均值、总均值 |

## 随机效应模型

### 固定因素与随机因素

若一个因素的水平是特意选定的、推断只限于这些水平，则该因素是**固定**的；温度等定量因素通常是固定的。若因素的水平是从可能水平的总体中随机抽取的（原材料批次、操作人员、机器），推断针对整个总体，则该因素是**随机**的。工业中的测量系统研究是典型应用。

:::{prf:assumption} 单因素随机效应模型
:label: asm-random-effects

$$
y_{ij} = \mu + \tau_i + \varepsilon_{ij},
\qquad i = 1,\ldots,a,\quad j = 1,\ldots,n,
$$

其中 $\tau_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_\tau^2)$，$\varepsilon_{ij} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$，且 $\tau_i$ 与 $\varepsilon_{ij}$ 相互独立。
:::

一个观测的方差是两个**方差分量**（variance component）之和：

$$
\Var(y_{ij}) = \sigma_\tau^2 + \sigma^2 ,
$$

同一处理下的两个观测相关，对 $j \ne j'$ 有 $\Cov(y_{ij}, y_{ij'}) = \sigma_\tau^2$。比值 $\sigma_\tau^2/(\sigma_\tau^2 + \sigma^2)$ 称为**组内相关系数**（intraclass correlation）。

### 假设

此时检验这几个特定处理的均值是否相等已无意义：试验中的处理只是一个样本，关心的是处理总体是否存在变异。

$$
\begin{gathered}
H_0:\ \sigma_\tau^2 = 0 \quad\text{对} \\
H_1:\ \sigma_\tau^2 > 0 .
\end{gathered}
$$

### 期望均方与检验

平方和、自由度和均方的计算与 {doc}`11-one-way-anova` 完全相同，恒等式 $\mathit{SS}_T = \mathit{SS}_{\mathrm{Treatments}} + \mathit{SS}_E$ 仍然成立，只有期望均方发生变化。

:::{prf:theorem} 随机效应模型的期望均方
:label: thm-random-ems

在 {prf:ref}`asm-random-effects` 下，

$$
\begin{gathered}
\E[\mathit{MS}_E] = \sigma^2, \\
\E[\mathit{MS}_{\mathrm{Treatments}}] = \sigma^2 + n\,\sigma_\tau^2 .
\end{gathered}
$$
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

在处理 $i$ 内，$y_{ij} - \bar{y}_{i\cdot} = \varepsilon_{ij} - \bar{\varepsilon}_{i\cdot}$ 不含 $\tau_i$，因此 $\mathit{SS}_E$ 的分布与固定效应模型中相同，$\E[\mathit{MS}_E] = \sigma^2$（{prf:ref}`thm-one-way-ems`）。处理均值 $\bar{y}_{i\cdot} = \mu + \tau_i + \bar{\varepsilon}_{i\cdot}$ 独立同分布，方差为 $\sigma_\tau^2 + \sigma^2/n$，而 $\mathit{MS}_{\mathrm{Treatments}} = n\sum_i(\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot})^2/(a-1)$ 是它们样本方差的 $n$ 倍。样本方差是方差的无偏估计（{prf:ref}`ex-sample-variance`），所以 $\E[\mathit{MS}_{\mathrm{Treatments}}] = n(\sigma_\tau^2 + \sigma^2/n)$。
:::

在 $H_0$ 下两个均方都是 $\sigma^2$ 的估计；此时处理均值独立同分布于正态分布，与 {prf:ref}`thm-one-way-distributions` 一样有

$$
F_0 = \frac{\mathit{MS}_{\mathrm{Treatments}}}{\mathit{MS}_E} \sim F(a-1,\ N-a).
$$

当 $F_0 > F_{1-\alpha;\,a-1,\,N-a}$ 时拒绝 $H_0$。检验统计量与固定效应模型相同，不同的是假设以及结论适用的范围。

### 估计方差分量

令均方等于其期望，得到**方差分析估计量**

$$
\begin{gathered}
\widehat{\sigma}^2 = \mathit{MS}_E, \\
\widehat{\sigma}_\tau^2 = \frac{\mathit{MS}_{\mathrm{Treatments}} - \mathit{MS}_E}{n} .
\end{gathered}
$$

两者都是无偏的。当 $\mathit{MS}_{\mathrm{Treatments}} < \mathit{MS}_E$ 时，$\widehat{\sigma}_\tau^2$ 可能为负；负的估计值通常报告为零，它表明处理间的变异很小，或者模型不合适。

若各组样本量 $n_1,\ldots,n_a$ **不等**，$N = \sum_i n_i$，则把 $\E[\mathit{MS}_{\mathrm{Treatments}}]$ 和 $\widehat{\sigma}_\tau^2$ 中的 $n$ 换成

$$
n_0 = \frac{1}{a-1}\left(N - \frac{\sum_{i=1}^a n_i^2}{N}\right).
$$

当所有 $n_i = n$ 时，$n_0 = n$。

## 区组设计

### 干扰因素

**干扰因素**（nuisance factor）可能影响响应，但其本身并不是研究对象，例如原材料批次、操作人员、试验设备、日期或班次、不同的试验单元。应当把它的变异从处理比较中剔除。

- 若干扰因素**已知且可控**，采用**区组**（blocking）。
- 若干扰因素**已知但不可控**且可以测量，可用**协方差分析**消除其影响。
- 若干扰因素**未知且不可控**（潜伏变量），则依靠**随机化**使其影响在各处理间平均地抵消。

**区组**（block）是干扰因素的一个水平，即一组比全体试验单元更同质的试验单元。区组之间的变异可能很大，但区组内部的变异应当较小。几个干扰来源可以合并为一个区组变量。

### 随机化完全区组设计

在**随机化完全区组设计**中，每个区组恰好包含每种处理一次（"完全"），并且在每个区组内分别随机安排试验顺序。例如，为比较硬度试验机的 $a = 4$ 种压头，每种压头在 $b = 4$ 块金属试样上各试验一次；试样就是区组，于是试样之间的差异被从试验误差中剔除。区组是对**随机化的限制**：试验在区组内随机化，而不是跨区组随机化。

:::{prf:assumption} 随机化完全区组设计
:label: asm-rcbd

$$
y_{ij} = \mu + \tau_i + \beta_j + \varepsilon_{ij},
\qquad i = 1,\ldots,a,\quad j = 1,\ldots,b,
$$

其中处理效应和区组效应是固定的，满足 $\sum_i\tau_i = 0$ 和 $\sum_j\beta_j = 0$，且 $\varepsilon_{ij} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$。
:::

处理均值为 $\mu_i = \frac1b\sum_j(\mu + \tau_i + \beta_j) = \mu + \tau_i$，关心的假设为

$$
\begin{gathered}
H_0:\ \mu_1 = \mu_2 = \cdots = \mu_a \quad\text{对} \\
H_1:\ \mu_i \ne \mu_j \text{ 至少对一对 } (i,j) \text{ 成立}.
\end{gathered}
$$

### 总变异的分解

记处理均值 $\bar{y}_{i\cdot} = \frac1b\sum_j y_{ij}$、区组均值 $\bar{y}_{\cdot j} = \frac1a\sum_i y_{ij}$、总均值 $\bar{y}_{\cdot\cdot} = \frac{1}{ab}\sum_i\sum_j y_{ij}$，写

$$
y_{ij} - \bar{y}_{\cdot\cdot}
= (\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot})
+ (\bar{y}_{\cdot j} - \bar{y}_{\cdot\cdot})
+ (y_{ij} - \bar{y}_{i\cdot} - \bar{y}_{\cdot j} + \bar{y}_{\cdot\cdot}) .
$$

两边平方后对 $i$ 和 $j$ 求和，由于每个分量对其与另一分量不共有的下标求和为零，交叉项都消失，于是

$$
\begin{aligned}
\sum_{i=1}^a\sum_{j=1}^b(y_{ij} - \bar{y}_{\cdot\cdot})^2
&= \underbrace{b\sum_{i=1}^a(\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot})^2}_{\mathit{SS}_{\mathrm{Treatments}}}
 + \underbrace{a\sum_{j=1}^b(\bar{y}_{\cdot j} - \bar{y}_{\cdot\cdot})^2}_{\mathit{SS}_{\mathrm{Blocks}}} \\
&\quad + \underbrace{\sum_{i=1}^a\sum_{j=1}^b(y_{ij} - \bar{y}_{i\cdot} - \bar{y}_{\cdot j} + \bar{y}_{\cdot\cdot})^2}_{\mathit{SS}_E},
\end{aligned}
$$

即 $\mathit{SS}_T = \mathit{SS}_{\mathrm{Treatments}} + \mathit{SS}_{\mathrm{Blocks}} + \mathit{SS}_E$，其自由度满足

$$
ab - 1 = (a-1) + (b-1) + (a-1)(b-1).
$$

### 期望均方与 $F$ 检验

各平方和除以其自由度得到均方，其期望为

$$
\begin{gathered}
\E[\mathit{MS}_{\mathrm{Treatments}}] = \sigma^2 + \frac{b}{a-1}\sum_{i=1}^a\tau_i^2, \\
\E[\mathit{MS}_{\mathrm{Blocks}}] = \sigma^2 + \frac{a}{b-1}\sum_{j=1}^b\beta_j^2, \\
\E[\mathit{MS}_E] = \sigma^2 .
\end{gathered}
$$

在 $H_0$ 下处理均方也是 $\sigma^2$ 的估计，且

$$
F_0 = \frac{\mathit{MS}_{\mathrm{Treatments}}}{\mathit{MS}_E} \sim F\bigl(a-1,\ (a-1)(b-1)\bigr).
$$

当 $F_0 > F_{1-\alpha;\,a-1,\,(a-1)(b-1)}$ 时拒绝 $H_0$。

| 变异来源 | 平方和 | 自由度 | 均方 | $F_0$ |
|---|---|---|---|---|
| 处理 | $\mathit{SS}_{\mathrm{Treatments}}$ | $a-1$ | $\mathit{SS}_{\mathrm{Treatments}}/(a-1)$ | $\mathit{MS}_{\mathrm{Treatments}}/\mathit{MS}_E$ |
| 区组 | $\mathit{SS}_{\mathrm{Blocks}}$ | $b-1$ | $\mathit{SS}_{\mathrm{Blocks}}/(b-1)$ | |
| 误差 | $\mathit{SS}_E$（相减得到） | $(a-1)(b-1)$ | $\mathit{SS}_E/\bigl((a-1)(b-1)\bigr)$ | |
| 总计 | $\mathit{SS}_T$ | $ab-1$ | | |

计算公式（$N = ab$）为

$$
\begin{gathered}
\mathit{SS}_T = \sum_{i=1}^a\sum_{j=1}^b y_{ij}^2 - \frac{y_{\cdot\cdot}^2}{N}, \\
\mathit{SS}_{\mathrm{Treatments}} = \frac1b\sum_{i=1}^a y_{i\cdot}^2 - \frac{y_{\cdot\cdot}^2}{N}, \\
\mathit{SS}_{\mathrm{Blocks}} = \frac1a\sum_{j=1}^b y_{\cdot j}^2 - \frac{y_{\cdot\cdot}^2}{N}, \\
\mathit{SS}_E = \mathit{SS}_T - \mathit{SS}_{\mathrm{Treatments}} - \mathit{SS}_{\mathrm{Blocks}} .
\end{gathered}
$$

区组是对随机化的限制，因此比值 $\mathit{MS}_{\mathrm{Blocks}}/\mathit{MS}_E$ 并不是区组效应的精确检验；其值较大只是说明划分区组是值得的。

### 多重比较

拒绝 $H_0$ 后，用 Fisher LSD（{doc}`11-one-way-anova`）比较处理均值对，每种处理有 $b$ 个观测，并使用 RCBD 的误差自由度：

$$
\mathrm{LSD} = t_{1-\alpha/2,\,(a-1)(b-1)}\sqrt{\frac{2\,\mathit{MS}_E}{b}} .
$$

当 $\lvert\bar{y}_{i\cdot} - \bar{y}_{j\cdot}\rvert > \mathrm{LSD}$ 时，判定 $\mu_i$ 与 $\mu_j$ 不同。

## 本章回顾
### 小结

- 在随机效应模型中，$\Var(y_{ij}) = \sigma_\tau^2 + \sigma^2$；$\E[\mathit{MS}_{\mathrm{Treatments}}] = \sigma^2 + n\sigma_\tau^2$，用 $F_0 = \mathit{MS}_{\mathrm{Treatments}}/\mathit{MS}_E$ 检验 $H_0:\sigma_\tau^2 = 0$。
- $\widehat{\sigma}^2 = \mathit{MS}_E$，$\widehat{\sigma}_\tau^2 = (\mathit{MS}_{\mathrm{Treatments}} - \mathit{MS}_E)/n$。
- 区组设计把已知、可控的干扰因素的变异从误差中剔除。
- 在 RCBD 中，$\mathit{SS}_T = \mathit{SS}_{\mathrm{Treatments}} + \mathit{SS}_{\mathrm{Blocks}} + \mathit{SS}_E$，处理用 $F(a-1,(a-1)(b-1))$ 检验。

### 常见错误

- 处理是随机样本时却检验这几个特定处理的均值是否相等；此时的假设针对 $\sigma_\tau^2$。
- 报告 $\sigma_\tau^2$ 的负估计值而不加说明。
- 忽略已知的干扰因素，使其变异增大 $\mathit{MS}_E$，降低处理检验的功效。
- 把 RCBD 当作单因素方差分析来分析，把区组的变异归入误差。

## 参考文献

- Montgomery, D. C. (2017). *Design and analysis of experiments* (9th ed.). Wiley.
- Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley.

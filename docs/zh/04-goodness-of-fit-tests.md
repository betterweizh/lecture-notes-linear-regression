# 拟合优度检验与 Q–Q 图

## 概述

本书的许多方法都假设数据或残差服从某种特定分布：线性模型中的正态误差，计数回归中的泊松计数。**拟合优度**（goodness-of-fit）方法检查样本是否与所假设的分布相符。本章介绍适用于离散数据或分组数据的 $\chi^2$ 拟合优度检验，适用于连续分布的 Kolmogorov–Smirnov 检验（单样本和双样本），以及分位数–分位数（Q–Q）图——检查回归残差正态性的标准图形方法。代码见 {doc}`23-correlation-and-goodness-of-fit-code`。 本章以分布函数（{doc}`01-probability-and-random-vectors`）以及 $\chi^2$ 分布和假设检验（{doc}`02-sampling-distributions-and-limit-theorems`）为基础。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 进行 $\chi^2$ 拟合优度检验，包括根据估计参数的个数调整自由度；
- 进行单样本和双样本 Kolmogorov–Smirnov 检验；
- 绘制正态 Q–Q 图及其参考线；
- 识别 Q–Q 图中常见的偏离正态的形态。
:::

## $\chi^2$ 拟合优度检验

### 设定

把随机变量 $X$ 的观测分为 $k$ 类（离散变量的取值，或连续变量的区间），观测频数为 $n_1,\ldots,n_k$，$n = \sum_i n_i$。所假设的分布 $F_0$ 给各类赋予概率 $p_1,\ldots,p_k$：

| 类别 | $1$ | $2$ | $\cdots$ | $k$ | 合计 |
|---|---|---|---|---|---|
| 观测频数 | $n_1$ | $n_2$ | $\cdots$ | $n_k$ | $n$ |
| 假设概率 | $p_1$ | $p_2$ | $\cdots$ | $p_k$ | $1$ |

假设为

$$
\begin{gathered}
H_0:\ F(x) = F_0(x) \quad\text{对} \\
H_1:\ F(x) \ne F_0(x).
\end{gathered}
$$

### 检验统计量

在 $H_0$ 下，第 $i$ 类的期望频数为 $np_i$。**Pearson $\chi^2$ 统计量**比较观测频数与期望频数：

$$
X^2 = \sum_{i=1}^k \frac{(n_i - np_i)^2}{np_i}
\ \overset{\text{近似}}{\sim}\ \chi^2(k - 1 - r),
$$ (eq-chi2-gof)

其中 $r$ 是由数据估计的 $F_0$ 的参数个数（$F_0$ 完全给定时 $r = 0$；估计 $\mu$ 和 $\sigma^2$ 的正态分布 $r = 2$；估计 $\lambda$ 的泊松分布 $r = 1$）。由于频数之和为 $n$，还要减去一个自由度。

$X^2$ 越大说明拟合越差，因此检验是单侧的：当 $X^2 \ge \chi^2_{1-\alpha,\,k-1-r}$ 时在水平 $\alpha$ 下拒绝 $H_0$。$X^2$ 较小与 $H_0$ 相容。

$\chi^2$ 近似要求期望频数足够大，常用规则是每一类都有 $np_i \ge 5$。不满足的类应与相邻的类合并，$k$ 也相应减小。

## Kolmogorov–Smirnov 检验

### 单样本检验

对连续的假设分布，分组会损失信息。**Kolmogorov–Smirnov（K–S）检验**比较阶梯函数形式的**经验分布函数**

$$
F_n(x) = \frac{\#\{i : x_i \le x\}}{n},
$$

与假设的连续分布函数 $F_0$，所用统计量是二者的最大纵向距离

$$
D_n = \sup_x\bigl\lvert F_n(x) - F_0(x)\bigr\rvert .
$$

由于 $F_n$ 在有序观测 $x_{(1)} \le \cdots \le x_{(n)}$ 处跳跃，上确界在跳跃点处或其左侧取得：

$$
D_n = \max_{1\le i\le n}\max\left\{\frac{i}{n} - F_0(x_{(i)}),\ F_0(x_{(i)}) - \frac{i-1}{n}\right\}.
$$

在 $H_0$ 下，$D_n$ 的分布与 $F_0$ 无关。当 $D_n$ 超过查表得到的临界值时拒绝 $H_0$；$n$ 较大时，当

$$
D_n \ge \frac{K_{1-\alpha}}{\sqrt{n}},
$$

时拒绝，其中 $K_{1-\alpha}$ 是 Kolmogorov 分布的 $(1-\alpha)$ 分位数（$K_{0.95} \approx 1.358$）。该检验假设 $F_0$ 完全给定；若其参数由同一组数据估计，标准临界值偏大（检验偏保守），应使用 Lilliefors 检验等修正版本。

### 双样本检验

检验大小为 $n$ 和 $m$ 的两个独立样本是否来自同一连续分布时，比较二者的经验分布函数 $F_n$ 与 $G_m$：

$$
D_{n,m} = \sup_x\bigl\lvert F_n(x) - G_m(x)\bigr\rvert .
$$

样本较大时，当 $D_{n,m} \ge K_{1-\alpha}\sqrt{(n+m)/(nm)}$ 时拒绝 $H_0$。

### 其他检验

K–S 统计量对分布中部的差异最敏感。**Anderson–Darling 检验**以 $1/\{F_0(1-F_0)\}$ 为权对 $F_n$ 与 $F_0$ 之差的平方加权，对尾部更敏感；**Shapiro–Wilk 检验**是专为正态性设计的功效较高的检验。

## 分位数–分位数图

### 定义

**Q–Q 图**通过把两个分布的分位数相互对照来比较它们：对每个概率 $p$，点 $(x_p, y_p)$ 满足

$$
\Prob(X \le x_p) = \Prob(Y \le y_p) = p .
$$

若两个分布相同，各点落在直线 $y = x$ 上；若二者只在位置和尺度上不同（$Y \overset{d}{=} a + bX$），各点落在直线 $y = a + bx$ 上。

### 正态 Q–Q 图

检查样本 $y_1,\ldots,y_n$ 是否来自正态分布时，把有序样本 $y_{(1)} \le \cdots \le y_{(n)}$ 对标准正态分位数作图：

$$
x_i = \Phi^{-1}\!\left(\frac{i - 0.5}{n}\right), \qquad i = 1,\ldots,n,
$$

其中 $(i-0.5)/n$ 称为**绘图位置**（plotting position），即第 $i$ 个次序统计量的经验概率，经调整后严格位于 $(0,1)$ 内。若样本来自均值为 $\mu$、标准差为 $\sigma$ 的正态分布，各点应接近直线

$$
y = \widehat{\mu} + \widehat{\sigma}\,x,
$$

其中 $\widehat{\mu}$ 和 $\widehat{\sigma}$ 为样本均值和样本标准差。（R 的 `qqline()` 则让直线通过第一和第三四分位数，受异常值的影响较小。）

### 解读 Q–Q 图

| 形态 | 解释 |
|---|---|
| 各点接近一条直线 | 与正态性相符 |
| S 形：中间平缓、两端陡峭 | 尾部比正态分布厚 |
| 反 S 形 | 尾部比正态分布薄 |
| 上凸或下凸的曲线 | 右偏或左偏分布 |
| 少数远离直线的孤立点 | 异常值 |

在回归中，残差的 Q–Q 图用于检查正态性假设（{doc}`13-prediction-and-model-assessment`、{doc}`11-one-way-anova`）。即使数据服从正态分布，小样本的图也会有一些波动，因此应依据图的整体形状来判断。

## 本章回顾
### 小结

- $\chi^2$ 拟合优度统计量 $X^2 = \sum_i(n_i - np_i)^2/(np_i)$ 在 $H_0$ 下近似服从 $\chi^2(k-1-r)$，值大时拒绝。
- K–S 统计量是经验分布函数与假设分布函数之间的最大距离；双样本版本比较两个经验分布函数。
- 把有序样本对 $\Phi^{-1}\bigl((i-0.5)/n\bigr)$ 作的正态 Q–Q 图，对正态数据接近一条直线；弯曲说明偏态或尾部非正态。

### 常见错误

- 期望频数低于 5 左右时不合并类别就使用 $\chi^2$ 检验。
- 忘记从 $\chi^2$ 检验的自由度中减去估计参数的个数。
- 用同一组数据估计 $F_0$ 的参数后，仍使用 K–S 检验的标准临界值。
- 把不显著的拟合优度检验当作所假设分布正确的证据；样本较小时，这些检验的功效很低。

## 参考文献

- Heumann, C., Schomaker, M., & Shalabh. (2016). *Introduction to statistics and data analysis: With exercises, solutions and applications in R*. Springer. <https://doi.org/10.1007/978-3-319-46162-5>
- Hollander, M., Wolfe, D. A., & Chicken, E. (2014). *Nonparametric statistical methods* (3rd ed.). Wiley.
- Kanji, G. K. (2006). *100 statistical tests* (3rd ed.). Sage.

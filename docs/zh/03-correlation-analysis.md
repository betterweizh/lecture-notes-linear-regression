# 相关分析

## 概述

在拟合回归模型之前，常常需要先度量两个变量之间关联的强弱。本章介绍度量连续变量线性关联的 Pearson 相关系数，度量单调关联、适用于有序数据的 Spearman 和 Kendall 秩相关系数，以及用于列联表的 Goodman–Kruskal $\gamma$。对每一种系数，给出定义、主要性质以及无关联的检验。代码见 {doc}`23-correlation-and-goodness-of-fit-code`。 本章以协方差与相关系数（{doc}`01-probability-and-random-vectors`）以及假设检验中的 $t$ 分布和正态分布（{doc}`02-sampling-distributions-and-limit-theorems`）为基础。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 计算并解释 Pearson 相关系数，并说明它与简单线性回归的关系；
- 用 $t$ 检验检验 $H_0:\rho = 0$，用 Fisher $z$ 变换检验 $H_0:\rho = \rho_0$，并构造 $\rho$ 的置信区间；
- 计算有结和无结情形下的 Spearman 与 Kendall 秩相关系数，并检验其显著性；
- 用 Goodman–Kruskal $\gamma$ 度量列联表中的关联。
:::

## 符号说明

| 符号 | 含义 |
|---|---|
| $(x_i, y_i)$, $i = 1,\ldots,n$ | $X$ 与 $Y$ 的成对观测 |
| $S_{xx}$, $S_{yy}$, $S_{xy}$ | 离差平方和与离差乘积和 |
| $\rho$, $r$ | 总体与样本 Pearson 相关系数 |
| $R_S$ | Spearman 秩相关系数 |
| $\tau$ | Kendall 秩相关系数 |
| $n_c$, $n_d$ | 一致对与不一致对的个数 |

## Pearson 相关系数

### 定义

$X$ 与 $Y$ 的总体相关系数为 $\rho = \Cov(X,Y)/\sqrt{\Var(X)\Var(Y)}$（{doc}`01-probability-and-random-vectors`）。它的样本对应量，即 **Pearson（或 Bravais–Pearson）相关系数**，为

$$
\begin{aligned}
r &= \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}
         {\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2\,\sum_{i=1}^n (y_i - \bar{y})^2}} \\
  &= \frac{S_{xy}}{\sqrt{S_{xx}S_{yy}}},
\end{aligned}
$$ (eq-pearson-r)

其中

$$
\begin{aligned}
S_{xx} &= \sum_{i=1}^n (x_i - \bar{x})^2, \\
S_{yy} &= \sum_{i=1}^n (y_i - \bar{y})^2, \\
S_{xy} &= \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}) \\
&= \sum_{i=1}^n x_iy_i - n\bar{x}\bar{y} .
\end{aligned}
$$

当两个变量都在连续尺度上测量、且关系近似线性时，适合使用它。

### 性质

- $-1 \le r \le 1$。当且仅当各点落在一条斜率为正（负）的直线上时，$r = 1$（$r = -1$）；符号给出关联的方向。
- $r$ 是对称的，$r(X,Y) = r(Y,X)$。
- $r$ 与测量单位无关：在变换 $x \mapsto a + bx$、$y \mapsto c + dy$（$bd > 0$）下不变（若 $bd < 0$ 则变号）。身高以米或厘米为单位得到的 $r$ 相同。
- $r$ 只度量**线性**关联。$r$ 接近零说明不存在线性关系；变量之间仍可能有很强的非线性关系，而且不相关的变量未必独立。

### 与简单线性回归的关系

在 $y$ 对 $x$ 的简单线性回归中（{doc}`08-ols-and-ml-estimation`），斜率估计值和决定系数（{doc}`13-prediction-and-model-assessment`）为

$$
\begin{aligned}
\widehat{\beta}_1 &= \frac{S_{xy}}{S_{xx}} \\
&= r\,\frac{s_y}{s_x}, \\
R^2 &= r^2 ,
\end{aligned}
$$

其中 $s_x$ 和 $s_y$ 为样本标准差。因此检验 $H_0:\rho = 0$ 等价于检验 $H_0:\beta_1 = 0$。

### 零相关的 $t$ 检验

设 $(X, Y)$ 服从二元正态分布。检验 $H_0:\rho = 0$ 时，使用

$$
T = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}} \sim t(n-2) \quad\text{在 } H_0 \text{ 下} .
$$ (eq-corr-t)

它就是简单线性回归中 $\widehat{\beta}_1$ 的 $t$ 统计量。当 $\lvert T\rvert > t_{1-\alpha/2,\,n-2}$ 时，在水平 $\alpha$ 下拒绝 $H_0$。

### Fisher $z$ 变换

检验 $H_0:\rho = \rho_0$（$\rho_0 \ne 0$）或构造 $\rho$ 的置信区间时，使用 **Fisher $z$ 变换**

$$
\begin{aligned}
z &= \operatorname{artanh}(r) \\
&= \frac12\ln\frac{1+r}{1-r} .
\end{aligned}
$$

对二元正态数据，$z$ 近似服从均值为 $\operatorname{artanh}(\rho)$、标准差为 $1/\sqrt{n-3}$ 的正态分布。因此在 $H_0:\rho = \rho_0$ 下，

$$
Z = \bigl(\operatorname{artanh}(r) - \operatorname{artanh}(\rho_0)\bigr)\sqrt{n-3}
\ \overset{\text{近似}}{\sim}\ \mathcal{N}(0,1),
$$ (eq-fisher-z)

$\rho$ 的近似 $100(1-\alpha)\%$ 置信区间为

$$
\tanh\!\left(\operatorname{artanh}(r) \pm \frac{z_{1-\alpha/2}}{\sqrt{n-3}}\right).
$$

## Spearman 秩相关系数

### 定义

把每个 $x_i$ 替换为它在 $x_1,\ldots,x_n$ 中的秩（最小值的秩为 1），每个 $y_i$ 替换为它在各 $y$ 中的秩。相同的值（结）取它们所占秩的平均值。**Spearman 秩相关系数** $R_S$ 就是秩的 Pearson 相关系数 {eq}`eq-pearson-r`：

$$
R_S = r\bigl(\operatorname{rank}(x), \operatorname{rank}(y)\bigr).
$$

无结时，记 $d_i = \operatorname{rank}(x_i) - \operatorname{rank}(y_i)$，它可简化为

$$
R_S = 1 - \frac{6\sum_{i=1}^n d_i^2}{n(n^2-1)} .
$$ (eq-spearman)

有结时，{eq}`eq-spearman` 只是近似，应按秩的定义计算。

### 性质

- $-1 \le R_S \le 1$。两组秩完全一致时 $R_S = 1$，一组秩恰为另一组的倒序时 $R_S = -1$。
- $R_S$ 在 $X$ 或 $Y$ 的任意严格递增变换下不变，因此它度量的是**单调**关联而不仅是线性关联，也可用于有序变量。
- 只要两个变量采用相同的排序方向，无论按升序还是降序赋秩，$R_S$ 都相同。

### 无关联的检验

在独立性假设 $H_0$ 下，$\E[R_S] = 0$，$\Var(R_S) = 1/(n-1)$。对大样本（大致 $n > 10$），

$$
Z = R_S\sqrt{n-1} \ \overset{\text{近似}}{\sim}\ \mathcal{N}(0,1).
$$

对小样本，$R_S$（或 $\sum_i d_i^2$）的精确临界值或 $p$ 值可查表或用置换法得到。

## Kendall 秩相关系数

### 一致对与不一致对

考虑两个观测 $(x_i, y_i)$ 和 $(x_j, y_j)$，$i < j$。这一对观测称为

- **一致对**（concordant pair），若 $x_i - x_j$ 与 $y_i - y_j$ 同号；
- **不一致对**（discordant pair），若二者异号；
- **结**（tied pair），若 $x_i = x_j$ 或 $y_i = y_j$。

记 $n_c$ 和 $n_d$ 分别为一致对和不一致对的个数，$n_0 = n(n-1)/2$ 为全部观测对的个数。

### 定义

无结时，**Kendall $\tau$** 是一致对与不一致对所占比例之差：

$$
\begin{aligned}
\tau &= \frac{n_c - n_d}{n_0} \\
     &= \frac{2}{n(n-1)}\sum_{i<j}\sign(x_i - x_j)\,\sign(y_i - y_j) .
\end{aligned}
$$ (eq-kendall)

和 $S = n_c - n_d$ 称为 **Kendall 得分**。与 $R_S$ 一样，$\tau$ 取值于 $[-1, 1]$，两组秩完全相同时为 $1$，完全相反时为 $-1$，$X$ 与 $Y$ 独立时接近零。

### 有结的情形：$\tau_A$、$\tau_B$ 与 $\tau_C$

结既不是一致对也不是不一致对，系数的几种版本对结的处理方式不同。

- **$\tau_A = (n_c - n_d)/n_0$** 不对结作调整；无结时它等于 {eq}`eq-kendall`。
- **$\tau_B$** 对分母作结的校正：

  $$
  \tau_B = \frac{n_c - n_d}{\sqrt{(n_0 - n_1)(n_0 - n_2)}},
  $$

  其中 $n_1 = \sum_k t_k(t_k-1)/2$，$n_2 = \sum_l u_l(u_l-1)/2$，$t_k$（$u_l$）为 $x$（$y$）中第 $k$（$l$）组结所含的相同值个数。它适用于方形列联表，也是大多数软件计算的版本。
- **$\tau_C$**（Stuart–Kendall）适用于 $r$ 行 $c$ 列的矩形列联表：

  $$
  \tau_C = \frac{2(n_c - n_d)}{n^2(m-1)/m}, \qquad m = \min(r, c).
  $$

### 无关联的检验

在独立性假设 $H_0$ 下且无结时，$\E[\tau] = 0$，$\Var(\tau) = 2(2n+5)/\bigl(9n(n-1)\bigr)$。对大样本（大致 $n > 10$），

$$
\begin{aligned}
Z &= \frac{\tau}{\sqrt{2(2n+5)/\bigl(9n(n-1)\bigr)}} \\
&= \frac{S}{\sqrt{n(n-1)(2n+5)/18}}
\ \overset{\text{近似}}{\sim}\ \mathcal{N}(0,1).
\end{aligned}
$$

对小样本，$S$ 的精确临界值有表可查；有结时须对 $S$ 的方差作校正。

## Goodman–Kruskal $\gamma$

对汇总为 $I\times J$ 列联表（频数 $n_{ij}$）的两个有序变量，观测对中一致对和不一致对的个数为

$$
\begin{gathered}
K = \sum_{i<k}\sum_{j<l} n_{ij}\,n_{kl}, \\
D = \sum_{i<k}\sum_{j>l} n_{ij}\,n_{kl},
\end{gathered}
$$

**Goodman–Kruskal $\gamma$** 为

$$
\gamma = \frac{K - D}{K + D} .
$$

结被忽略，因此 $-1 \le \gamma \le 1$，其符号给出关联的方向，绝对值越大关联越强。由于去掉了结，$\lvert\gamma\rvert \ge \lvert\tau_B\rvert$。

## 系数的选择

| 系数 | 度量 | 数据 | 对异常值稳健 |
|---|---|---|---|
| Pearson $r$ | 线性关联 | 连续数据，检验时需近似二元正态 | 否 |
| Spearman $R_S$ | 单调关联 | 连续或有序数据 | 是 |
| Kendall $\tau$ | 单调关联（基于观测对） | 连续或有序数据，小样本 | 是 |
| Goodman–Kruskal $\gamma$ | 有序列联表中的关联 | 有序的分组数据 | 是 |

## 本章回顾
### 小结

- $r = S_{xy}/\sqrt{S_{xx}S_{yy}}$ 度量线性关联；在简单线性回归中 $R^2 = r^2$。
- $T = r\sqrt{n-2}/\sqrt{1-r^2} \sim t(n-2)$ 检验 $\rho = 0$；Fisher $z = \operatorname{artanh}(r)$ 的标准误为 $1/\sqrt{n-3}$，用于检验 $\rho = \rho_0$ 和构造置信区间。
- Spearman $R_S$ 是秩的 Pearson 相关系数；Kendall $\tau$ 比较一致对与不一致对。二者都度量单调关联，并对异常值稳健。
- Goodman–Kruskal $\gamma$ 度量有序列联表中的关联。

### 常见错误

- 因为 $r \approx 0$ 就断定两个变量独立；$r$ 只能发现线性关联。
- 把显著的相关解读为因果关系的证据。
- 用 $t$ 检验 {eq}`eq-corr-t` 检验 $H_0:\rho = \rho_0 \ne 0$；它只在 $\rho_0 = 0$ 时有效。
- 在结很多时使用无结公式 {eq}`eq-spearman`。

## 参考文献

- Heumann, C., Schomaker, M., & Shalabh. (2016). *Introduction to statistics and data analysis: With exercises, solutions and applications in R*. Springer. <https://doi.org/10.1007/978-3-319-46162-5>
- Hollander, M., Wolfe, D. A., & Chicken, E. (2014). *Nonparametric statistical methods* (3rd ed.). Wiley.
- Kanji, G. K. (2006). *100 statistical tests* (3rd ed.). Sage.
- Wikipedia contributors. (n.d.). Kendall rank correlation coefficient. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>

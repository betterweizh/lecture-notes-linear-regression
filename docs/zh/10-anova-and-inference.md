# 方差分析与统计推断

## 概述

响应变量的总变异可以分解为回归能够解释的部分和残差部分。这种**方差分析**（analysis of variance, ANOVA）是检验整体模型以及一组系数的 $F$ 检验的基础。结合检验单个系数的 $t$ 检验，就可以回答"哪些预测变量对模型有贡献"这一问题。 本章以 {doc}`09-ols-properties-and-geometry` 中的正交性和抽样分布结论（{prf:ref}`prop-orthogonality`、{prf:ref}`thm-ols-distribution`）以及 $t$、$\chi^2$ 和 $F$ 分布为基础。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 对含截距和不含截距的模型证明分解式 $\mathit{SS}_T = \mathit{SS}_R + \mathit{SS}_E$；
- 构造方差分析表，并说明各平方和的自由度；
- 对单个系数进行 $t$ 检验并构造置信区间；
- 用 $F$ 检验检验整体回归和部分系数。
:::

## 符号说明

| 符号 | 含义 |
|---|---|
| $\bar{y}$ | 响应的样本均值 $\frac1n\sum_i y_i$ |
| $\mathit{SS}_T$ | 总平方和 |
| $\mathit{SS}_R$ | 回归平方和 |
| $\mathit{SS}_E$ | 残差（误差）平方和 |
| $\mathit{MS}_R$, $\mathit{MS}_E$ | 均方：平方和除以其自由度 |
| $c_{jj}$ | $(\bX\T\bX)^{-1}$ 的第 $j$ 个对角元，$j=0,\ldots,k$ |
| $t_{q,\nu}$ | 自由度为 $\nu$ 的 $t$ 分布的 $q$ 分位数 |
| $F_{q;\nu_1,\nu_2}$ | 自由度为 $(\nu_1,\nu_2)$ 的 $F$ 分布的 $q$ 分位数 |

## 方差分解

### 含截距的模型

:::{prf:theorem} 平方和分解
:label: thm-ss-decomposition

对含截距的线性模型，

$$
\underbrace{\sum_{i=1}^n (y_i-\bar{y})^2}_{\mathit{SS}_T}
= \underbrace{\sum_{i=1}^n (\widehat{y}_i-\bar{y})^2}_{\mathit{SS}_R}
+ \underbrace{\sum_{i=1}^n (y_i-\widehat{y}_i)^2}_{\mathit{SS}_E} .
$$ (eq-ss-decomposition)

即：总平方和 = 回归平方和 + 残差平方和。
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

将 $y_i - \bar{y}$ 写成 $(y_i - \widehat{y}_i) + (\widehat{y}_i - \bar{y})$ 并展开平方：

$$
\sum_{i=1}^n (y_i-\bar{y})^2
= \sum_{i=1}^n (y_i-\widehat{y}_i)^2 + \sum_{i=1}^n (\widehat{y}_i-\bar{y})^2
+ 2\sum_{i=1}^n (\widehat{y}_i-\bar{y})(y_i-\widehat{y}_i).
$$

交叉项为零：

$$
\begin{aligned}
\sum_{i=1}^n (\widehat{y}_i-\bar{y})\,e_i &= \underbrace{\widehat{\by}\T\be}_{=0} - \bar{y}\underbrace{\sum_{i=1}^n e_i}_{=0} \\
&= 0,
\end{aligned}
$$

这由 {prf:ref}`prop-orthogonality` 得到，其中第二个等式用到了截距。
:::

常用的计算公式为

$$
\begin{aligned}
\mathit{SS}_T &= \sum_{i=1}^n y_i^2 - n\bar{y}^2 \\
             &= \by\T\by - \tfrac{1}{n}\Bigl(\sum_{i=1}^n y_i\Bigr)^2, \\
\mathit{SS}_R &= \widehat{\bbeta}\T\bX\T\by - \tfrac{1}{n}\Bigl(\sum_{i=1}^n y_i\Bigr)^2, \\
\mathit{SS}_E &= (\by-\bX\widehat{\bbeta})\T(\by-\bX\widehat{\bbeta}) \\
             &= \by\T\by - \widehat{\bbeta}\T\bX\T\by .
\end{aligned}
$$

最后一个等式用到正规方程 $\widehat{\bbeta}\T\bX\T\bX\widehat{\bbeta} = \widehat{\bbeta}\T\bX\T\by$。$\mathit{SS}_T$、$\mathit{SS}_R$、$\mathit{SS}_E$ 的自由度分别为 $n-1$、$p-1 = k$ 和 $n-p$，它们与平方和一样满足相加关系。

### 不含截距的模型

对不含截距的模型

$$
y_i = \beta_1x_{i1} + \beta_2x_{i2} + \cdots + \beta_kx_{ik} + \varepsilon_i,
\qquad i = 1,\ldots,n,
$$

拟合超平面经过原点而不是均值点 $(\bar{\bx},\bar{y})$，残差之和也不一定为零。因此分解使用未中心化的平方和：

$$
\underbrace{\sum_{i=1}^n y_i^2}_{\mathit{SS}_T}
= \underbrace{\sum_{i=1}^n \widehat{y}_i^{\,2}}_{\mathit{SS}_R}
+ \underbrace{\sum_{i=1}^n (y_i-\widehat{y}_i)^2}_{\mathit{SS}_E} .
$$ (eq-ss-decomposition-noint)

交叉项 $2\sum_i \widehat{y}_i e_i = 2\,\widehat{\by}\T\be$ 由 {prf:ref}`prop-orthogonality` 知为零，而这一点并不需要截距。用矩阵表示，

$$
\begin{aligned}
\mathit{SS}_T &= \by\T\by, \\
\mathit{SS}_R &= \widehat{\bbeta}\T\bX\T\bX\widehat{\bbeta} \\
&= \widehat{\bbeta}\T\bX\T\by, \\
\mathit{SS}_E &= \by\T\by - \widehat{\bbeta}\T\bX\T\by ,
\end{aligned}
$$

自由度分别为 $n$、$k$ 和 $n-k$（此时 $p = k$）。

:::{warning}
由于不含截距模型的 $\mathit{SS}_T$ 未经中心化，其报告的 $R^2$ 不能与含截距模型的 $R^2$ 相比较。
:::

## 方差分析表

### 一元线性回归

对 $y_i = \beta_0 + \beta_1x_i + \varepsilon_i$（$S_{xx}$ 和 $S_{xy}$ 的定义见 {doc}`08-ols-and-ml-estimation`）：

| 变异来源 | 平方和 | 自由度 | 均方 |
|---|---|---|---|
| 回归 | $\mathit{SS}_R = \widehat{\beta}_1S_{xy}$ | $1$ | $\mathit{MS}_R = \mathit{SS}_R/1$ |
| 误差 | $\mathit{SS}_E = \mathit{SS}_T - \widehat{\beta}_1S_{xy}$ | $n-2$ | $\mathit{MS}_E = \mathit{SS}_E/(n-2)$ |
| 总计 | $\mathit{SS}_T$ | $n-1$ | |

均方的期望为

$$
\begin{gathered}
\E[\mathit{MS}_E] = \sigma^2, \\
\E[\mathit{MS}_R] = \sigma^2 + \beta_1^2S_{xx},
\end{gathered}
$$

因此当 $\beta_1 \ne 0$ 时，$\mathit{MS}_R/\mathit{MS}_E$ 倾向于大于 1。

### 含截距的多元线性回归

| 变异来源 | 平方和 | 自由度 | 均方 |
|---|---|---|---|
| 回归 | $\sum_{i}(\widehat{y}_i-\bar{y})^2$ | $k$ | $\mathit{MS}_R = \mathit{SS}_R/k$ |
| 误差 | $\sum_{i}(y_i-\widehat{y}_i)^2$ | $n-k-1$ | $\mathit{MS}_E = \mathit{SS}_E/(n-k-1)$ |
| 总计 | $\sum_{i}(y_i-\bar{y})^2$ | $n-1$ | |

### 不含截距的多元线性回归

| 变异来源 | 平方和 | 自由度 | 均方 |
|---|---|---|---|
| 回归 | $\sum_{i}\widehat{y}_i^{\,2}$ | $k$ | $\mathit{MS}_R = \mathit{SS}_R/k$ |
| 误差 | $\sum_{i}(y_i-\widehat{y}_i)^2$ | $n-k$ | $\mathit{MS}_E = \mathit{SS}_E/(n-k)$ |
| 总计 | $\sum_{i}y_i^2$ | $n$ | |

### 平方和的分布

:::{prf:theorem} 方差分析表中的分布
:label: thm-anova-distributions

对含截距的模型，在 {prf:ref}`asm-classical` 下：

1. $\mathit{SS}_E/\sigma^2 \sim \chi^2(n-p)$（由 {prf:ref}`thm-ols-distribution`）；
2. $\mathit{SS}_R$、$\mathit{SS}_E$ 和 $\bar{y}$ 相互独立；
3. 若 $H_0:\beta_1 = \cdots = \beta_k = 0$ 成立，则 $\mathit{SS}_R/\sigma^2 \sim \chi^2(k)$。
:::

第 3 条只在原假设下成立；当某个斜率不为零时，$\mathit{SS}_R/\sigma^2$ 服从非中心 $\chi^2$ 分布。对不含截距的模型，相应结论为 $\mathit{SS}_E/\sigma^2 \sim \chi^2(n-k)$，以及在 $H_0$ 下 $\mathit{SS}_R/\sigma^2 \sim \chi^2(k)$。

## 单个系数的推断

### $t$ 检验

检验系数 $\beta_j$ 是否等于给定值 $\beta_{j,0}$（通常为 $0$）：

$$
\begin{gathered}
H_0 :\ \beta_j = \beta_{j,0} \quad\text{对} \\
H_1 :\ \beta_j \ne \beta_{j,0}, \qquad j = 0,1,\ldots,k,
\end{gathered}
$$

使用统计量

$$
\begin{aligned}
T_j &= \frac{\widehat{\beta}_j - \beta_{j,0}}{\mathrm{SE}(\widehat{\beta}_j)} \\
&= \frac{\widehat{\beta}_j - \beta_{j,0}}{\widehat{\sigma}\sqrt{c_{jj}}} \ \sim\ t(n-p) \quad\text{（在 } H_0 \text{ 下）},
\end{aligned}
$$ (eq-t-test)

其中 $\mathrm{SE}(\widehat{\beta}_j) = \sqrt{\widehat{\sigma}^2c_{jj}}$ 是 $\widehat{\beta}_j$ 的**标准误**（standard error），即估计量标准差的估计值。$t$ 分布来自 {prf:ref}`thm-ols-distribution`：分子除以 $\sigma\sqrt{c_{jj}}$ 后服从标准正态分布，它与 $\widehat{\sigma}^2$ 独立，且 $(n-p)\widehat{\sigma}^2/\sigma^2 \sim \chi^2(n-p)$。

在显著性水平 $\alpha$ 下，若观测值 $t_j$ 满足 $|t_j| > t_{1-\alpha/2,\,n-p}$，则拒绝 $H_0$。

### 置信区间

$\beta_j$ 的 $100(1-\alpha)\%$ 置信区间为

$$
\widehat{\beta}_j - t_{1-\alpha/2,\,n-p}\,\mathrm{SE}(\widehat{\beta}_j)
\ \le\ \beta_j\ \le\
\widehat{\beta}_j + t_{1-\alpha/2,\,n-p}\,\mathrm{SE}(\widehat{\beta}_j).
$$

## 整体 $F$ 检验

**整体 $F$ 检验**判断回归是否解释了任何变异，即是否至少有一个斜率不为零：

$$
\begin{gathered}
H_0 :\ \beta_1 = \beta_2 = \cdots = \beta_k = 0 \quad\text{对} \\
H_1 :\ \text{至少有一个 } j \in \{1,\ldots,k\} \text{ 使 } \beta_j \ne 0 .
\end{gathered}
$$

原假设只涉及斜率，不检验截距 $\beta_0$。由 {prf:ref}`thm-anova-distributions`，

$$
\begin{aligned}
F &= \frac{\mathit{MS}_R}{\mathit{MS}_E} \\
&= \frac{\mathit{SS}_R/(p-1)}{\mathit{SS}_E/(n-p)} \ \sim\ F(p-1,\ n-p) \quad\text{（在 } H_0 \text{ 下）}.
\end{aligned}
$$ (eq-overall-f)

当观测值 $f$ 超过 $F_{1-\alpha;\,p-1,\,n-p}$ 时，在水平 $\alpha$ 下拒绝 $H_0$。

## 检验部分系数

### 额外平方和

将系数划分为 $\bbeta = (\bbeta_1\T, \bbeta_2\T)\T$，其中 $\bbeta_1$ 包含待检验的 $r$ 个系数，$\bbeta_2$ 包含其余 $p-r$ 个系数（含截距），并相应地划分 $\bX = [\bX_1,\ \bX_2]$。

对拟合后的模型，定义**未校正回归平方和**

$$
\begin{aligned}
\mathit{SS}_R(\bbeta) &= \widehat{\bbeta}\T\bX\T\by \\
&= \mathit{SS}_R + \frac{1}{n}\Bigl(\sum_{i=1}^n y_i\Bigr)^2 .
\end{aligned}
$$

记 $\widehat{\bbeta}_2^{(0)}$ 为简化模型 $\by = \bX_2\bbeta_2 + \bvarepsilon$ 的 OLS 估计量。给定 $\bbeta_2$ 时 $\bbeta_1$ 的**额外平方和**（extra sum of squares）是在简化模型中加入 $\bX_1$ 后回归平方和的增量：

$$
\begin{aligned}
\mathit{SS}_R(\bbeta_1 \mid \bbeta_2)
&= \mathit{SS}_R(\bbeta) - \mathit{SS}_R(\bbeta_2) \\
 &= \widehat{\bbeta}\T\bX\T\by - \bigl(\widehat{\bbeta}_2^{(0)}\bigr)\T\bX_2\T\by \\
&= \underbrace{\Bigl[\mathit{SS}_R(\bbeta) - \tfrac{1}{n}\bigl(\textstyle\sum_i y_i\bigr)^2\Bigr]}_{\text{全模型的 } \mathit{SS}_R}
 - \underbrace{\Bigl[\mathit{SS}_R(\bbeta_2) - \tfrac{1}{n}\bigl(\textstyle\sum_i y_i\bigr)^2\Bigr]}_{\text{简化模型的 } \mathit{SS}_R} .
\end{aligned}
$$

等价地，$\mathit{SS}_R(\bbeta_1\mid\bbeta_2)$ 等于加入 $\bX_1$ 后 $\mathit{SS}_E$ 的减少量。

### 偏 $F$ 检验

为检验

$$
\begin{gathered}
H_0 :\ \bbeta_1 = \bzero \quad\text{对} \\
H_1 :\ \bbeta_1 \ne \bzero ,
\end{gathered}
$$

使用

$$
\begin{aligned}
F_r &= \frac{\mathit{SS}_R(\bbeta_1\mid\bbeta_2)/r}{\mathit{SS}_E/(n-p)} \\
&= \frac{\mathit{SS}_R(\bbeta_1\mid\bbeta_2)/r}{\mathit{MS}_E} \ \sim\ F(r,\ n-p) \quad\text{（在 } H_0 \text{ 下）},
\end{aligned}
$$ (eq-partial-f)

其中分子有 $r$ 个自由度（每个被检验的系数一个），$\mathit{SS}_E$ 来自全模型，自由度为 $n-p$。若观测值超过 $F_{1-\alpha;\,r,\,n-p}$，则拒绝 $H_0$，认为 $\bbeta_1$ 中至少有一个系数不为零，即在其他变量给定时，$\bX_1$ 中至少有一个预测变量有显著贡献。当 $r = 1$ 时，偏 $F$ 检验等价于双侧 $t$ 检验 {eq}`eq-t-test`，因为 $F_1 = T_j^2$。

:::{prf:example} 序贯平方和
:label: ex-sequential-ss

对含四个预测变量的模型，回归平方和可以通过逐个加入预测变量来分解：

$$
\begin{aligned}
\mathit{SS}_R &= \mathit{SS}_R(\beta_1,\beta_2,\beta_3,\beta_4 \mid \beta_0) \\
&= \mathit{SS}_R(\beta_1\mid\beta_0)
 + \mathit{SS}_R(\beta_2\mid\beta_1,\beta_0)
 + \mathit{SS}_R(\beta_3\mid\beta_2,\beta_1,\beta_0)
 + \mathit{SS}_R(\beta_4\mid\beta_3,\beta_2,\beta_1,\beta_0).
\end{aligned}
$$

每一项衡量的是在先进入的变量给定时某个预测变量的贡献，因此各项取决于变量进入的顺序。
:::

## 本章回顾
### 小结

- 含截距时，$\mathit{SS}_T = \mathit{SS}_R + \mathit{SS}_E$，自由度满足 $n-1 = k + (n-p)$；不含截距时，未中心化平方和按 $n = k + (n-k)$ 分解。
- $T_j = (\widehat{\beta}_j-\beta_{j,0})/\mathrm{SE}(\widehat{\beta}_j) \sim t(n-p)$ 用于检验单个系数。
- $F = \mathit{MS}_R/\mathit{MS}_E \sim F(p-1,n-p)$ 用于联合检验全部斜率。
- 偏 $F$ 检验通过额外平方和比较嵌套模型。

### 常见错误

- 无条件地声称 $\mathit{SS}_R/\sigma^2 \sim \chi^2(k)$；这只在 $H_0$ 下成立。
- 把显著的整体 $F$ 检验理解为"所有系数都不为零"。它只说明至少有一个斜率不为零。
- 解释序贯平方和时忽略了它们依赖于预测变量的进入顺序。
- 将含截距模型与不含截距模型的 $R^2$ 相比较。

## 参考文献

- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An introduction to statistical learning: With applications in R*. Springer. <https://doi.org/10.1007/978-1-4614-7138-7>
- Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley.
- He, Z. [何志坚]. (n.d.). *数理统计讲义* [Lecture notes on mathematical statistics]. <https://bookdown.org/hezhijian/book/>
- Zaiontz, C. (n.d.). *Multiple regression without intercept*. Real Statistics Using Excel. <https://real-statistics.com/multiple-regression/multiple-regression-without-intercept>

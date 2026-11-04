# 变量选择与子集选择

## 概述

当候选预测变量很多时，分析者必须决定保留哪些变量。包含较多预测变量的模型可以更细致地描述数据，但更难解释，也可能过拟合；包含较少预测变量的模型使用方便，但可能遗漏重要效应。本章汇总比较候选模型的准则，以及寻找较优子集的搜索算法，并演示如何在 R 中实现。 本章以偏 $F$ 检验（{doc}`10-anova-and-inference`）以及 $R^2$、调整 $R^2$ 和 PRESS（{doc}`13-prediction-and-model-assessment`）为基础。下一章 {doc}`16-ridge-and-lasso` 介绍惩罚回归，它是离散子集选择的一种连续替代方法。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 陈述变量选择问题及其假设；
- 计算并比较 $R^2_p$、$R^2_{\mathrm{adj}}(p)$、$\mathit{MS}_E(p)$、Mallows $C_p$、AIC、AICc 和 BIC；
- 描述最优子集、向前、向后和逐步选择方法及其计算量。
:::

## 选择问题

### 动机

- 建模时，从众多可能的预测变量中选出相关变量非常重要。
- 变量选择就是对候选变量进行筛选，得到"最优"子集。
- 需要在细致但变量多的模型与简洁易用但变量少的模型之间权衡。
- 不存在唯一的"最优"模型：不同准则和方法可能选出不同的子集。应避免过拟合（overfitting），并且仍需要专业领域的判断。

### 问题表述

从候选集合 $\{x_1, x_2, \ldots, x_K\}$ 中选出"最优"的预测变量子集，假设：

- 只有一个响应 $y$ 和 $K$ 个候选预测变量；
- 每个模型都包含截距 $\beta_0$；
- 子集模型含 $p$ 个系数，即截距加 $p-1$ 个预测变量，$p = 1, 2, \ldots, K+1$。

需要回答两个问题：

1. 用什么**准则**衡量子集的好坏；
2. 用什么**搜索方法**找到较优子集，而不必拟合所有子集。

下文中 $\mathit{SS}_E(p)$、$\mathit{SS}_R(p)$ 和 $R_p^2$ 表示由含 $p$ 个系数的子集模型计算的量，$\mathit{SS}_T$ 与子集无关。

## 选择准则

### $R^2$ 准则

$$
\begin{aligned}
R_p^2 &= \frac{\mathit{SS}_R(p)}{\mathit{SS}_T} \\
&= 1 - \frac{\mathit{SS}_E(p)}{\mathit{SS}_T}.
\end{aligned}
$$

随着 $p$ 增大，$\mathit{SS}_T$ 不变，$\mathit{SS}_R(p)$ 增大，$\mathit{SS}_E(p)$ 减小，所以最大的模型总有最大的 $R_p^2$。因此该准则需借助图形使用：按 $p$ 对模型排序，画出 $R_p^2$ 关于 $p$ 的**上包络线**（upper envelope），选择继续增加预测变量带来的改进已很小的那个点。

### 调整 $R^2$ 准则

$$
R^2_{\mathrm{adj}}(p) = 1 - \frac{n-1}{n-p}\bigl(1-R_p^2\bigr).
$$

随着 $p$ 增大，$1-R_p^2$ 和 $n-p$ 都减小，所以 $R^2_{\mathrm{adj}}(p)$ 不一定增大，取决于哪一项的作用占优。**准则：**选择 $R^2_{\mathrm{adj}}(p)$ 最大的模型。

### 残差均方准则

$$
\mathit{MS}_E(p) = \frac{\mathit{SS}_E(p)}{n-p}.
$$

随着 $p$ 增大，$\mathit{SS}_E(p)$ 和 $n-p$ 都减小。通常 $\mathit{MS}_E(p)$ 的下包络线先随 $p$ 增大而下降，达到最小值后可能上升。**准则：**选择 $\mathit{MS}_E(p)$ 最小的模型。

:::{prf:proposition} $\mathit{MS}_E$ 准则与调整 $R^2$ 准则等价
:label: prop-mse-adjr2

使 $\mathit{MS}_E(p)$ 最小的子集模型同时使 $R^2_{\mathrm{adj}}(p)$ 最大。
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

$$
\begin{aligned}
R^2_{\mathrm{adj}}(p) &= 1 - \frac{n-1}{n-p}\cdot\frac{\mathit{SS}_E(p)}{\mathit{SS}_T} \\
&= 1 - \frac{n-1}{\mathit{SS}_T}\cdot\frac{\mathit{SS}_E(p)}{n-p} \\
&= 1 - \frac{n-1}{\mathit{SS}_T}\,\mathit{MS}_E(p).
\end{aligned}
$$

由于 $(n-1)/\mathit{SS}_T$ 是与子集无关的正常数，$R^2_{\mathrm{adj}}(p)$ 是 $\mathit{MS}_E(p)$ 的减函数。
:::

### Mallows $C_p$

**Mallows $C_p$** 定义为

$$
C_p = \frac{\mathit{SS}_E(p)}{\widehat{\sigma}^2} - n + 2p,
$$ (eq-mallows-cp)

其中 $\mathit{SS}_E(p)$ 是含 $p$ 个系数的子集模型的残差平方和，$\widehat{\sigma}^2$ 由包含全部 $K$ 个候选预测变量的**全模型**估计。另一种形式为

$$
\begin{aligned}
C_p^{*} &= \frac{1}{n}\bigl(\mathit{SS}_E(p) + 2p\,\widehat{\sigma}^2\bigr) \\
&= \frac{\widehat{\sigma}^2}{n}\bigl(C_p + n\bigr).
\end{aligned}
$$

两种形式由同一子集取得最小值。偏差很小的子集模型满足 $C_p \approx p$。

### 赤池信息准则（AIC）

#### 一般定义

对含 $d$ 个待估参数、极大似然值为 $\widehat{\mathcal{L}}$ 的模型，**赤池信息准则**（Akaike information criterion）为

$$
\mathrm{AIC} = 2d - 2\ln\widehat{\mathcal{L}} .
$$

数值越小越好。

#### 线性回归中的 AIC

对正态线性模型，将 {doc}`08-ols-and-ml-estimation` 中的 ML 估计量代入对数似然 {eq}`eq-normal-loglik`，并取 $\widehat{\sigma}^2_{\mathrm{ML}} = \mathit{SS}_E/n$，得

$$
\begin{aligned}
\ln\widehat{\mathcal{L}}
&= -\frac{n}{2}\ln\bigl(2\pi\widehat{\sigma}^2_{\mathrm{ML}}\bigr)
   - \frac{1}{2\widehat{\sigma}^2_{\mathrm{ML}}}\sum_{i=1}^n(y_i-\widehat{y}_i)^2 \\
 &= -\frac{n}{2}\ln\bigl(2\pi\widehat{\sigma}^2_{\mathrm{ML}}\bigr) - \frac{n}{2} \\
&= -\frac{n}{2}\ln\!\left(\frac{\mathit{SS}_E}{n}\right)
   \underbrace{-\ \frac{n}{2}\ln(2\pi) - \frac{n}{2}}_{\text{常数}} .
\end{aligned}
$$

含 $p$ 个系数的线性模型有 $d = p + 1$ 个参数（系数和 $\sigma^2$）。略去对同一数据上所有模型都相同的项，得到常用形式

$$
\mathrm{AIC} = n\ln\!\left(\frac{\mathit{SS}_E}{n}\right) + 2p .
$$ (eq-aic-linear)

取 $d = p$ 还是 $d = p+1$ 只会使每个模型的 AIC 改变同一个常数，不影响排序。

:::{note}
软件报告的 AIC 可能包含常数项。例如 `statsmodels` 用完整的对数似然 `res.llf` 计算 AIC，因此其数值与 {eq}`eq-aic-linear` 相差一个常数。同一数据上不同模型之间的 AIC 差值不受影响。
:::

#### AIC 与 Mallows $C_p$

若将误差方差固定为全模型估计值 $\widehat{\sigma}^2$，而不是对每个子集分别估计，则 $-2\ln\widehat{\mathcal{L}} = n\ln(2\pi\widehat{\sigma}^2) + \mathit{SS}_E/\widehat{\sigma}^2$，从而

$$
\begin{aligned}
\mathrm{AIC}
&= 2p + n\ln(2\pi\widehat{\sigma}^2) + \frac{\mathit{SS}_E(p)}{\widehat{\sigma}^2} \\
&= n\ln(2\pi\widehat{\sigma}^2) + C_p + n \\
 &= n\ln(2\pi\widehat{\sigma}^2) + \frac{n}{\widehat{\sigma}^2}\,C_p^{*} .
\end{aligned}
$$

在这种情况下，AIC 与 Mallows $C_p$ 选出同一个模型。

#### 修正 AIC

对小样本，**修正 AIC**（corrected AIC; Hurvich & Tsai, 1989）加入二阶偏差修正：

$$
\mathrm{AICc} = \mathrm{AIC} + \frac{2d^2 + 2d}{n - d - 1}.
$$

当 $n\to\infty$ 时，AICc 趋于 AIC。

### 贝叶斯信息准则（BIC）

**贝叶斯信息准则**（Bayesian information criterion）为

$$
\mathrm{BIC} = d\ln(n) - 2\ln\widehat{\mathcal{L}},
$$

其中 $d$ 为待估参数个数，$n$ 为观测数。对线性回归，略去常数后

$$
\mathrm{BIC} = n\ln\!\left(\frac{\mathit{SS}_E}{n}\right) + p\ln(n).
$$

由于 $n \ge 8$ 时 $\ln(n) > 2$，BIC 对增加系数的惩罚比 AIC 更重，倾向于选择更小的模型。

### 准则汇总

| 准则 | 最优模型 | 是否惩罚模型规模 |
|---|---|---|
| $R_p^2$ | 最大（借助包络线图） | 否 |
| $R^2_{\mathrm{adj}}(p)$ | 最大 | 是 |
| $\mathit{MS}_E(p)$ | 最小 | 是（与 $R^2_{\mathrm{adj}}$ 等价） |
| $R^2_{\mathrm{pred}}$ / PRESS | 最大 / 最小 | 通过交叉验证间接惩罚 |
| Mallows $C_p$ | 最小，且 $C_p \approx p$ | 是 |
| AIC、AICc | 最小 | 是 |
| BIC | 最小 | 是，惩罚更强 |

## 搜索算法

### 最优子集选择

最优子集选择（best subset selection）拟合 $K$ 个候选变量的所有组合，共 $2^K$ 个模型。例如，当 $K = 2$（$x_1$ 和 $x_2$）时，共有 $2^2 = 4$ 个模型。

:::{prf:algorithm} 最优子集选择
:label: alg-best-subset

1. 记 $\mathcal{M}_0$ 为只含截距的**零模型**，它对每个观测都预测样本均值 $\bar{y}$。
2. 对 $m = 1, 2, \ldots, K$：
   1. 拟合恰好含 $m$ 个预测变量的全部 $\binom{K}{m}$ 个模型；
   2. 从中选出 $\mathit{SS}_E$ 最小（等价地，$R^2$ 最大）的模型，记为 $\mathcal{M}_m$。
3. 用交叉验证预测误差，或考虑了模型规模的准则（如 $C_p$、AIC、BIC、调整 $R^2$ 或预测 $R^2$），从 $\mathcal{M}_0, \mathcal{M}_1, \ldots, \mathcal{M}_K$ 中选出一个模型。
:::

第 3 步不能使用 $\mathit{SS}_E$ 或 $R^2$：随着预测变量个数增加，$\mathcal{M}_0,\ldots,\mathcal{M}_K$ 的 $\mathit{SS}_E$ 单调下降，$R^2$ 单调上升。

### 向前逐步选择

:::{prf:algorithm} 向前逐步选择
:label: alg-forward

1. 从零模型 $\mathcal{M}_0$ 开始。
2. 对 $m = 0, 1, \ldots, K-1$：
   1. 考虑在 $\mathcal{M}_m$ 的基础上增加一个预测变量的全部 $K-m$ 个模型；
   2. 选出其中最优者（$\mathit{SS}_E$ 最小或 $R^2$ 最大），记为 $\mathcal{M}_{m+1}$。
3. 按 {prf:ref}`alg-best-subset` 第 3 步，从 $\mathcal{M}_0, \ldots, \mathcal{M}_K$ 中选出一个模型。
:::

向前选择只需拟合 $1 + K(K+1)/2$ 个模型，而不是 $2^K$ 个。

### 向后逐步选择

:::{prf:algorithm} 向后逐步选择
:label: alg-backward

1. 从包含全部 $K$ 个预测变量的全模型 $\mathcal{M}_K$ 开始。
2. 对 $m = K, K-1, \ldots, 1$：
   1. 考虑从 $\mathcal{M}_m$ 中删去一个预测变量的全部 $m$ 个模型，每个模型含 $m-1$ 个预测变量；
   2. 选出其中最优者（$\mathit{SS}_E$ 最小或 $R^2$ 最大），记为 $\mathcal{M}_{m-1}$。
3. 按 {prf:ref}`alg-best-subset` 第 3 步，从 $\mathcal{M}_0, \ldots, \mathcal{M}_K$ 中选出一个模型。
:::

向后选择要求 $n > K$，以便能够拟合全模型。

### 混合（逐步）选择

**逐步选择**（stepwise selection，也称混合选择）结合了两个方向：从不含预测变量开始，每一步像向前选择那样加入最有用的预测变量；每次加入后，再像向后选择那样删去不再改善拟合的预测变量。这种方法更接近最优子集选择，同时保留了逐步方法计算量小的优点。

#### 基于偏 $F$ 检验的逐步选择

逐步回归的经典版本可能是应用最广的变量选择方法，它的每一步决策都基于 {doc}`10-anova-and-inference` 中的偏 $F$ 检验：

- 设定两个阈值：进入阈值 $\alpha_{\text{in}}$ 和剔除阈值 $\alpha_{\text{out}}$，并取 $\alpha_{\text{in}} \le \alpha_{\text{out}}$，以免刚进入的预测变量立即被剔除；
- 若预测变量 $x_j$ 的偏 $F$ 统计量的 $p$ 值小于 $\alpha_{\text{in}}$，则**加入**；
- 若预测变量 $x_j$ 的偏 $F$ 统计量的 $p$ 值大于 $\alpha_{\text{out}}$，则**剔除**。

:::{prf:algorithm} 基于偏 $F$ 检验的逐步选择
:label: alg-stepwise-f

1. **开始。** 加入与 $y$ 相关性最强的单个预测变量，记为 $x_1$。
2. **加入。** 对其余每个候选变量 $x_j$，$j = 2,\ldots,K$，计算

   $$
   F_j = \frac{\mathit{SS}_R(\beta_j\mid\beta_1,\beta_0)/1}{\mathit{MS}_E(x_j, x_1)},
   $$

   其中 $\mathit{MS}_E(x_j,x_1)$ 为包含 $x_1$ 和 $x_j$ 的模型的残差均方。设 $j^{*}$ 为 $F_j$ 最大（$p$ 值最小）的候选变量；若该 $p$ 值小于 $\alpha_{\text{in}}$，则加入 $x_{j^{*}}$。
3. **剔除。** 在 $x_{j^{*}}$ 已进入模型的情况下，检查是否仍需保留 $x_1$：

   $$
   F_1 = \frac{\mathit{SS}_R(\beta_1\mid\beta_{j^{*}},\beta_0)/1}{\mathit{MS}_E(x_1,x_{j^{*}})}.
   $$

   若 $F_1$ 的 $p$ 值超过 $\alpha_{\text{out}}$，则剔除 $x_1$。当有多个预测变量可能被剔除时，对当前模型中的每个预测变量 $x_j$ 计算

   $$
   F_j = \frac{\mathit{SS}_R(\beta_j\mid \text{模型中其余全部系数})}{\mathit{MS}_E},
   $$

   若 $F_j$ 最小者的 $p$ 值超过 $\alpha_{\text{out}}$，则将其剔除。
4. **重复**第 2、3 步，直到没有预测变量可以加入或剔除。之前被剔除的预测变量仍可再次进入。
:::

每个 $F_j$ 的自由度取决于当前模型中预测变量的个数。

该方法有两种常见的简化版本：

**向前选择**
: 只加入、不剔除预测变量，因此忽略了新加入变量对已有变量的影响。

**向后剔除**
: 从全部 $K$ 个候选变量开始，每次剔除 $p$ 值最大且超过 $\alpha_{\text{out}}$ 的预测变量；当没有变量满足剔除条件时停止。

## 实现

实现上述方法的 R 函数 `regsubsets()`（`leaps` 包）和 `step()` 的用法见 {doc}`25-linear-regression-r`。

## 说明与建议

- 对最终模型进行充分性检验：残差、强影响点和过拟合（{doc}`13-prediction-and-model-assessment`）。
- 尝试多种建模和变量选择方法，检查结论是否一致。
- 若候选变量不太多，推荐使用最优子集选择，因为它保证在所选准则下找到每种规模的最优模型。
- 逐步方法不能保证得到"最优"模型。

## 本章回顾
### 小结

- 变量选择需要一个准则和一种搜索方法。
- $R^2_{\mathrm{adj}}$、$\mathit{MS}_E$、$C_p$、AIC、AICc、BIC 和 PRESS 考虑了模型规模；$R^2$ 没有。
- 最优子集选择需拟合 $2^K$ 个模型；向前和向后选择大约只需 $K^2/2$ 个。

### 常见错误

- 用 $R^2$ 或 $\mathit{SS}_E$ 比较不同规模的模型。
- 把所选模型的 $p$ 值当作事先设定模型的 $p$ 值来报告；变量选择会使它们过于乐观。
- 将选出的子集视为唯一的真实模型。

## 参考文献

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction* (2nd ed.). Springer. <https://doi.org/10.1007/978-0-387-84858-7>
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An introduction to statistical learning: With applications in R*. Springer. <https://doi.org/10.1007/978-1-4614-7138-7>
- Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley.
- Hurvich, C. M., & Tsai, C.-L. (1989). Regression and time series model selection in small samples. *Biometrika, 76*(2), 297–307. <https://doi.org/10.1093/biomet/76.2.297>
- Soch, J., & The Book of Statistical Proofs contributors. (n.d.). *The book of statistical proofs*. <https://statproofbook.github.io/>
- Wikipedia contributors. (n.d.). Akaike information criterion. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Akaike_information_criterion>
- Wikipedia contributors. (n.d.). Bayesian information criterion. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Bayesian_information_criterion>
- Wikipedia contributors. (n.d.). Mallows's Cp. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Mallows%27s_Cp>

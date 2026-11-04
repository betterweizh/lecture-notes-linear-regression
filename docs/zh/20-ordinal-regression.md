# 有序回归

## 概述

**有序**响应变量的类别有自然的顺序，但类别之间没有有意义的距离，例如"不太可能" < "有些可能" < "很可能"。把这类响应当作数值变量处理，忽略了类别间距未知这一事实；把它当作无序变量处理（如 {doc}`21-multinomial-choice-models`），又忽略了顺序信息。**累积联系模型**（cumulative link model）通过对"落在某类别及其以下"的概率建模来直接利用顺序信息。它推广了二元 logit 和 probit 回归，属于 {doc}`17-generalized-linear-models` 的框架。 本章以联系函数（{doc}`17-generalized-linear-models`）和二元 logit 与 probit 模型（{doc}`18-logistic-regression`）为基础。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 定义有序响应的累积概率、累积优势和切点；
- 写出有序 logit（比例优势）模型和有序 probit 模型；
- 将有序 logit 模型的系数解释为累积优势比；
- 写出用于估计模型的对数似然函数。
:::

## 问题设定

设 $Y$ 是有 $J$ 个有序类别 $1 < 2 < \cdots < J$ 的有序响应，$\bx$ 是**不含**截距列的预测变量向量（下文的切点起到截距的作用）。类别 $j$ 的**累积概率**为

$$
\Prob(Y \le j \mid \bx), \qquad j = 1,\ldots,J-1,
$$

且 $\Prob(Y \le J\mid\bx) = 1$。落在类别 $j$ 及以下的**累积优势**（cumulative odds）为

$$
\frac{\Prob(Y \le j\mid\bx)}{\Prob(Y > j\mid\bx)}, \qquad j = 1,\ldots,J-1 .
$$

## 累积联系模型

累积联系模型设定

$$
F^{-1}\bigl(\Prob(Y \le j\mid\bx)\bigr) = \theta_j - \bbeta\T\bx,
\qquad j = 1,\ldots,J-1,
$$ (eq-cumulative-link)

其中 $F$ 是连续的累积分布函数（CDF），**切点**（cut-point，也称阈值）满足 $\theta_1 < \theta_2 < \cdots < \theta_{J-1}$。对每个 $j$ 使用同一个系数向量 $\bbeta$，只有切点不同。负号是一种常用约定：$\beta_l$ 为正时，概率向较高类别移动。

### 有序 logit 模型

累积优势的对数就是累积概率的 **logit**：

$$
\ln\!\left(\frac{\Prob(Y \le j\mid\bx)}{\Prob(Y > j\mid\bx)}\right)
= \logit\bigl(\Prob(Y \le j\mid\bx)\bigr).
$$

**有序 logistic 回归**模型取 $F$ 为 logistic 分布函数：

$$
\logit\bigl(\Prob(Y \le j\mid\bx)\bigr) = \theta_j - \bbeta\T\bx,
\qquad j = 1,\ldots,J-1 .
$$ (eq-ordinal-logit)

对两个预测变量向量 $\bx_a$ 和 $\bx_b$，累积对数优势之差对每个 $j$ 都等于 $-\bbeta\T(\bx_a - \bx_b)$。因此累积优势比与 $j$ 无关，这就是该模型又称**比例优势模型**（proportional odds model）的原因：对每个切点 $j$，$x_l$ 增加一个单位都使 $Y \le j$ 的优势乘以 $e^{-\beta_l}$（等价地，使 $Y > j$ 的优势乘以 $e^{\beta_l}$）。

### 有序 probit 模型

**有序 probit** 模型取 $F = \Phi$，即标准正态分布函数：

$$
\Phi^{-1}\bigl(\Prob(Y \le j\mid\bx)\bigr) = \theta_j - \bbeta\T\bx,
\qquad j = 1,\ldots,J-1 .
$$

它来自潜变量模型 $Y^{*} = \bbeta\T\bx + \varepsilon$，$\varepsilon \sim \mathcal{N}(0,1)$，当 $\theta_{j-1} < Y^{*} \le \theta_j$ 时观测到的类别为 $Y = j$。若将正态误差换成标准 logistic 误差，就得到有序 logit 模型。

## 对数似然函数

由 {eq}`eq-cumulative-link`，每个类别的概率是相邻累积概率之差：

$$
\Prob(Y = j\mid\bx) = F(\theta_j - \bbeta\T\bx) - F(\theta_{j-1} - \bbeta\T\bx),
\qquad j = 1,\ldots,J,
$$

约定 $\theta_0 = -\infty$，$\theta_J = +\infty$。对观测 $(\bx_i, y_i)$，$i = 1,\ldots,n$，对数似然函数为

$$
\ell(\bbeta,\boldsymbol\theta)
= \sum_{i=1}^n\sum_{j=1}^J \mathbb{I}(y_i = j)\,
  \ln\bigl[F(\theta_j - \bbeta\T\bx_i) - F(\theta_{j-1} - \bbeta\T\bx_i)\bigr],
$$

其中 $\mathbb{I}(\cdot)$ 为示性函数，$\boldsymbol\theta = (\theta_1,\ldots,\theta_{J-1})\T$。对 probit 模型，$F = \Phi$ 为标准正态分布函数，即对 $Z\sim\mathcal{N}(0,1)$ 有 $\Phi(x) = \Prob(Z \le x)$；对 logit 模型，$F(x) = 1/(1+e^{-x})$。参数通过数值方法极大化 $\ell$ 得到。

## 本章回顾
### 小结

- 累积联系模型用有序切点 $\theta_j$ 和共同的 $\bbeta$ 描述 $\Prob(Y\le j\mid\bx)$。
- 使用 logit 联系时，$e^{-\beta_l}$ 是在每个切点上都相同的累积优势比（比例优势）。
- 使用 probit 联系时，模型对应于在阈值处被切分的正态潜变量。
- 类别概率是相邻累积概率之差，由此得到对数似然函数。

### 常见错误

- 在有切点的同时又在 $\bx$ 中加入截距，导致模型不可识别。
- 忽略符号约定。不同软件的约定不同：有的使用 $\theta_j - \bbeta\T\bx$（如本书），有的使用 $\theta_j + \bbeta\T\bx$。
- 不加检验就假定比例优势成立；当某个预测变量在不同切点上的效应不同时，需要更灵活的模型。

## 参考文献

- Agresti, A. (2013). *Categorical data analysis* (3rd ed.). Wiley.
- Frank, E., & Hall, M. (2001). A simple approach to ordinal classification. In L. De Raedt & P. Flach (Eds.), *Machine learning: ECML 2001* (pp. 145–156). Springer. <https://doi.org/10.1007/3-540-44795-4_13>
- UCLA Statistical Methods and Data Analytics. (n.d.). *Ordinal logistic regression: R data analysis examples*. <https://stats.oarc.ucla.edu/r/dae/ordinal-logistic-regression/>
- Wikipedia contributors. (n.d.). Ordinal regression. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Ordinal_regression>

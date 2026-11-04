# 泊松回归与计数回归

## 概述

计数型响应（例如单位时间或单位空间内的事件数）是非负整数，其方差通常随均值增大而增大。泊松回归（Poisson regression）是处理这类数据的 GLM，它使用泊松分布和对数联系函数（{doc}`17-generalized-linear-models`）。本章介绍该模型及其解释、基于似然的检验和诊断，以及当数据违背其假设时的三种扩展：处理过度离散的拟泊松模型和负二项模型，以及处理过多零值的零膨胀泊松模型。 本章以指数族、对数联系和偏差（{doc}`17-generalized-linear-models`）为基础。本章的 Python 代码见 {doc}`27-glm-python`。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 陈述泊松回归的假设，并与线性回归的假设相比较；
- 将系数解释为对数率比，并用偏移项处理暴露量；
- 用偏差下降检验比较嵌套模型，用 Wald 检验检验单个系数；
- 计算 Pearson 残差和偏差残差，并评价拟合优度；
- 识别过度离散，并在拟泊松、负二项和零膨胀模型之间做出选择。
:::

## 泊松分布

在**泊松过程**中，事件相互独立地发生，某一区间内的事件数只取决于区间的长度或大小。在这样一个区间内的事件数 $Y$ 服从**泊松分布**：

$$
\Prob(Y = y) = \frac{e^{-\lambda}\lambda^y}{y!}, \qquad y = 0,1,2,\ldots,
$$

其中 $\lambda > 0$ 是单位时间或空间内的均值（期望计数）。泊松分布的一个基本性质是均值等于方差：

$$
\begin{aligned}
\E[Y] &= \Var(Y) \\
&= \lambda .
\end{aligned}
$$

## 泊松回归模型

### 假设

- **泊松响应。** 响应 $Y_i$ 是单位时间或空间内的计数，服从均值为 $\lambda_i$ 的泊松分布。
- **独立性。** 各观测相互独立。
- **均值等于方差。** $\E[Y_i] = \Var(Y_i) = \lambda_i$。
- **线性。** 平均发生率的（自然）对数是预测变量的线性函数：

$$
\begin{aligned}
\ln(\lambda_i) &= \beta_0 + \beta_1x_{i1} + \beta_2x_{i2} + \cdots + \beta_kx_{ik} \\
&= \bx_i\T\bbeta, \qquad i = 1,\ldots,n .
\end{aligned}
$$ (eq-poisson-model)

用 GLM 的语言，这就是使用典则对数联系的泊松分布，$\eta_i = \ln\mu_i$，$\mu_i = \lambda_i$。

### 与线性回归的比较

在线性回归中：

- 在 $x$ 的每个水平上，响应近似服从正态分布；
- 在 $x$ 的每个水平上，$y$ 的方差 $\sigma^2$ 相同；
- 平均响应 $\mu_{y\mid x}$ 落在一条直线上。

在泊松回归中：

- 当 $\lambda$ 较小时，响应分布明显右偏，小值很多、大值很少；随着 $\lambda$ 增大，分布越来越接近正态分布；
- 由于方差等于均值，响应的变异随均值增大而增大；
- 均值 $\lambda_{y\mid x}$ 落在一条曲线而非直线上，但其对数落在一条直线上。

### 偏移项与暴露量

计数常常是在大小不同的单位上记录的，例如不同的时间长度、人口规模或面积。若 $E_i$ 为观测 $i$ 的**暴露量**（exposure），则自然的建模对象是发生率 $\lambda_i/E_i$：

$$
\ln\!\left(\frac{\lambda_i}{E_i}\right) = \bx_i\T\bbeta
\quad\Longleftrightarrow\quad
\ln(\lambda_i) = \ln(E_i) + \bx_i\T\bbeta .
$$

项 $o_i = \ln(E_i)$ 称为**偏移项**（offset）：它以固定为 1 的系数进入线性预测子，而不需要估计。因此，指定暴露量 $E_i$ 等价于指定偏移项 $\ln E_i$。

## 解释：率比

考虑只在 $x_1$ 上相差一个单位的两个预测变量向量。由 {eq}`eq-poisson-model`，

$$
\left.\begin{gathered}
\ln(\lambda_{x_1}) = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots \\
\ln(\lambda_{x_1+1}) = \beta_0 + \beta_1(x_1+1) + \beta_2x_2 + \cdots
\end{gathered}\right\}
\quad\Longrightarrow\quad
\frac{\lambda_{x_1+1}}{\lambda_{x_1}} = e^{\beta_1}.
$$

在其他预测变量不变时，$x_1$ 每增加一个单位，平均计数乘以 $e^{\beta_1}$，即**率比**（rate ratio，也称相对风险）；等价地，均值变化 $100\,(e^{\beta_1}-1)\%$。

## 估计

对 $n$ 个独立观测，似然函数为

$$
\begin{aligned}
\mathcal{L}(\bbeta) &= \prod_{i=1}^n \Prob(Y_i = y_i) \\
&= \prod_{i=1}^n \frac{e^{-\lambda_i}\lambda_i^{y_i}}{y_i!},
\end{aligned}
$$

其中每个 $\lambda_i = \exp(\bx_i\T\bbeta)$ 取决于观测 $i$ 的预测变量。对数似然为

$$
\begin{aligned}
\ell(\bbeta) &= \ln\mathcal{L}(\bbeta) \\
&= \sum_{i=1}^n \bigl[-\lambda_i + y_i\ln\lambda_i - \ln(y_i!)\bigr] \\
&= \sum_{i=1}^n \bigl[-\exp(\bx_i\T\bbeta) + y_i\,\bx_i\T\bbeta - \ln(y_i!)\bigr].
\end{aligned}
$$ (eq-poisson-loglik)

$\ln(y_i!)$ 与 $\bbeta$ 无关。由于对数联系是典则联系，$\ell$ 严格凹，可用 {doc}`17-generalized-linear-models` 中的算法求极大似然估计。

## 假设检验

### 单个系数的 Wald 检验

为检验

$$
\begin{gathered}
H_0 :\ \beta_j = 0 \quad\text{对} \\
H_1 :\ \beta_j \ne 0,
\end{gathered}
$$

使用

$$
Z = \frac{\widehat{\beta}_j - 0}{\mathrm{SE}(\widehat{\beta}_j)} \ \overset{\text{近似}}{\sim}\ \mathcal{N}(0,1) \quad\text{（在 } H_0 \text{ 下）}.
$$

$|Z|$ 较大说明系数不为零。

### 偏差下降检验

偏差下降检验（drop-in-deviance test）比较两个**嵌套**模型，即较小模型的所有项都出现在较大模型中：

$$
\begin{gathered}
H_0 :\ \text{简化模型已足够（全模型中多出的系数为零）}, \\
H_1 :\ \text{需要较大的（全）模型}.
\end{gathered}
$$

拟合两个模型并计算

$$
\begin{aligned}
\text{偏差下降} &= D_{\text{reduced}} - D_{\text{full}} \\
&= -2\bigl[\ell(\widehat{\bbeta}_{\text{reduced}}) - \ell(\widehat{\bbeta}_{\text{full}})\bigr],
\end{aligned}
$$

其中 $D$ 为残差偏差 {eq}`eq-deviance`。由于饱和模型的对数似然相互抵消，偏差下降等于似然比统计量。在 $H_0$ 下，

$$
D_{\text{reduced}} - D_{\text{full}} \ \overset{\text{近似}}{\sim}\ \chi^2(d),
$$

其中 $d$ 为两个模型系数个数之差。偏差下降较大时支持较大的模型。

## 拟合优度

**残差偏差**衡量拟合值与观测计数之间的差距。若模型合适且计数不太小，则

$$
D \ \overset{\text{近似}}{\sim}\ \chi^2(n-p),
$$

其中 $n-p$ 为残差自由度，$p$ 为系数个数（含截距）。偏差远大于 $n-p$ 表明拟合不足。

## 残差

**Pearson 残差**将原始残差除以响应标准差的估计：

$$
r_{P,i} = \frac{y_i - \widehat{\lambda}_i}{\sqrt{\widehat{\lambda}_i}},
$$

其中 $\widehat{\lambda}_i$ 为观测 $i$ 的拟合均值。若模型合适，大多数 Pearson 残差应落在 $-2$ 与 $2$ 之间。

**偏差残差**是每个观测对偏差贡献的带符号平方根：

$$
r_{D,i} = \sign(y_i-\widehat{\lambda}_i)\sqrt{2\left[y_i\ln\!\left(\frac{y_i}{\widehat{\lambda}_i}\right) - (y_i-\widehat{\lambda}_i)\right]},
$$

约定 $y_i = 0$ 时 $y_i\ln(y_i/\widehat{\lambda}_i) = 0$，其中

$$
\sign(x) = \begin{cases} 1, & x > 0, \\ 0, & x = 0, \\ -1, & x < 0. \end{cases}
$$

残差偏差等于偏差残差的平方和，$D = \sum_i r_{D,i}^2$。

## 过度离散

**过度离散**（overdispersion）是指响应的变异大于模型所隐含的变异。在泊松模型下，同一组相似观测的响应均值与方差应大致相等；但实际中方差常常大于均值。忽略过度离散会导致标准误偏小、检验过于宽松。

两种常用的处理方法：

- 用估计的离散因子放大标准误（拟泊松）；
- 使用负二项回归模型。

### 拟泊松

用下式估计**离散参数** $\phi$（有时称为 Pearson 离散度，Hilbe, 2011）：

$$
\widehat{\phi} = \frac{\sum_{i=1}^n r_{P,i}^2}{n-p},
$$

其中 $n$ 为观测数，$p$ 为系数个数，$n-p$ 为残差自由度。若模型合适，$\sum_i r_{P,i}^2$ 近似服从 $\chi^2(n-p)$，因此 $\widehat{\phi}$ 应接近 1；存在过度离散时它会明显大于 1。

拟泊松（quasi-Poisson）方法保留泊松估计值，但将方差乘以 $\widehat{\phi}$，使标准误大于似然方法给出的值：

$$
\mathrm{SE}_Q(\widehat{\beta}_j) = \sqrt{\widehat{\phi}}\ \mathrm{SE}(\widehat{\beta}_j).
$$

下标 $Q$ 表示"拟"：用 $\widehat{\phi}$ 放大方差是一种临时性的调整，并不对应完整的似然函数（"拟似然"）。

### 泊松推断与拟泊松推断的比较

| | 泊松 | 拟泊松 |
|---|---|---|
| 估计值 | $\widehat{\beta}$ | $\widehat{\beta}$ |
| 标准误 | $\mathrm{SE}(\widehat{\beta})$ | $\mathrm{SE}_Q(\widehat{\beta}) = \sqrt{\widehat{\phi}}\,\mathrm{SE}(\widehat{\beta})$ |
| Wald 型检验统计量 | $Z = \widehat{\beta}/\mathrm{SE}(\widehat{\beta})$ | $t = \widehat{\beta}/\mathrm{SE}_Q(\widehat{\beta})$ |
| 置信区间 | $\widehat{\beta} \pm z_{1-\alpha/2}\,\mathrm{SE}(\widehat{\beta})$ | $\widehat{\beta} \pm t_{1-\alpha/2,\,n-p}\,\mathrm{SE}_Q(\widehat{\beta})$ |
| 偏差下降检验 | $\chi^2 = D_{\text{reduced}} - D_{\text{full}}$ | $F = \dfrac{(D_{\text{reduced}} - D_{\text{full}})/d}{\widehat{\phi}}$ |

拟泊松一栏中，$t$ 近似服从自由度为 $n-p$ 的 $t$ 分布；$F$ 近似服从 $F$ 分布，分子自由度为 $d$（系数个数之差），分母自由度为 $n-p$（全模型的残差自由度）。

### 负二项回归

负二项模型有以下优点：

- 除均值外引入了第二个参数，使模型更灵活；
- 与拟泊松方法不同，它给出了明确的似然函数；
- 负二项随机变量取非负整数值，与计数数据一致。

#### 负二项分布

若 $Y$ 为成功概率为 $p_s$ 的独立伯努利试验中第 $r$ 次成功之前的失败次数，则 $Y$ 服从**负二项分布**：

$$
\Prob(Y = y) = \binom{y + r - 1}{r - 1}(1-p_s)^y\,p_s^{\,r},
\qquad y = 0,1,2,\ldots,
$$

其中 $0 < p_s \le 1$。其均值和方差为

$$
\begin{aligned}
\E[Y] &= \mu \\
&= \frac{r(1-p_s)}{p_s}, \\
\Var(Y) &= \frac{r(1-p_s)}{p_s^2} \\
&= \mu + \frac{\mu^2}{r}.
\end{aligned}
$$

**与泊松分布的关系。** 若 $Y\mid\lambda \sim \mathrm{Poisson}(\lambda)$，而 $\lambda$ 服从形状参数为 $r$、尺度参数为 $(1-p_s)/p_s$ 的伽马分布，则 $Y \sim \mathrm{NegBin}(r, p_s)$。因此，负二项分布是均值在观测之间随机变化的泊松分布。额外的方差 $\mu^2/r$ 衡量过度离散；当 $r\to\infty$ 时它趋于零，所以 $r$ 越小，过度离散越严重。

#### 回归模型

负二项回归保留 {eq}`eq-poisson-model` 中的对数联系 $\ln\mu_i = \bx_i\T\bbeta$，并将泊松方差替换为 $\Var(Y_i) = \mu_i + \mu_i^2/r$。系数和 $r$（或离散参数 $\alpha = 1/r$）用极大似然法估计，系数仍保持率比的解释。

## 零膨胀泊松模型

有些计数数据的零值比泊松分布所允许的多，因为零值来自两个过程：一个总是产生零（"结构性"零），另一个是可能偶然产生零的泊松过程。

### 零膨胀泊松分布

观测以概率 $\pi$ 为结构性零，以概率 $1-\pi$ 来自 $\mathrm{Poisson}(\lambda)$：

$$
\Prob(Y = y) =
\begin{cases}
\pi + (1-\pi)\,e^{-\lambda}, & y = 0, \\[4pt]
(1-\pi)\,\dfrac{\lambda^y e^{-\lambda}}{y!}, & y = 1,2,3,\ldots
\end{cases}
$$

零膨胀泊松（zero-inflated Poisson, ZIP）变量的均值和方差为

$$
\begin{gathered}
\E[Y] = (1-\pi)\,\lambda, \\
\Var(Y) = \lambda(1-\pi)(1+\pi\lambda).
\end{gathered}
$$

### 零膨胀泊松回归

ZIP 回归模型是一个含两个线性预测子的混合（潜类别）模型：

$$
\begin{gathered}
\text{计数部分：} \qquad  \ln(\lambda_i) = \beta_0 + \beta_1x_{i1} + \beta_2x_{i2} + \cdots, \\
\text{零部分：} \qquad  \logit(\pi_i) = \gamma_0 + \gamma_1z_{i1} + \gamma_2z_{i2} + \cdots,
\end{gathered}
$$

其中 $\pi_i$ 是观测 $i$ 属于结构性零组的概率，零部分的预测变量 $z_{i1}, z_{i2},\ldots$ 可以与计数部分不同。零部分描述观测属于"总为零"组还是"计数"组。

### Vuong 检验

普通的似然比检验要求模型嵌套，但泊松模型与 ZIP 模型在通常意义下并不嵌套。**Vuong 检验**比较两个非嵌套模型的预测概率，常用于在二者之间做出选择。

## 本章回顾
### 小结

- 泊松回归建模 $\ln\E[Y_i] = \bx_i\T\bbeta$，并假设均值等于方差。
- $e^{\beta_j}$ 是率比；偏移项 $\ln E_i$ 用于处理暴露量。
- 嵌套模型用偏差下降检验比较；单个系数用 Wald 检验。
- 过度离散用拟泊松标准误或负二项回归处理；过多零值用零膨胀模型处理。

### 常见错误

- 把 $e^{\beta_1}$ 理解为加性变化，或理解为"变化 $e^{\beta_1}\%$"；它是均值的乘性变化。
- 把 $\ln(E_i)$ 作为需要估计系数的普通预测变量，而不是偏移项。
- 把残差偏差当作 $-2\ell(\widehat{\bbeta})$；只有偏差之差才等于 $-2\ell$ 之差。
- 忽略过度离散，导致标准误偏小。

## 参考文献

- Hilbe, J. M. (2011). *Negative binomial regression* (2nd ed.). Cambridge University Press. <https://doi.org/10.1017/CBO9780511973420>
- Roback, P., & Legler, J. (2021). *Beyond multiple linear regression: Applied generalized linear models and multilevel models in R*. CRC Press. <https://bookdown.org/roback/bookdown-BeyondMLR/>
- StataCorp. (n.d.). *Stata documentation* [Software documentation]. <https://www.stata.com/features/documentation/>

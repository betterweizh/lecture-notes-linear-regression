# 多项选择模型

## 概述

多项选择模型描述决策者如何从有限个无序备选方案中选出一个，例如出行者选择交通方式。这类模型建立在**随机效用最大化**（random utility maximization）之上：每个方案的效用由取决于可观测属性的系统部分和分析者无法观测的随机部分组成，决策者选择效用最高的方案。对随机部分的不同假设会得到不同的模型。本章推导多项 logit（MNL）模型和多项 probit（MNP）模型，并说明如何估计和评价它们。 本章以极大似然和 logit 联系（{doc}`17-generalized-linear-models`）以及离散响应的潜变量观点（{doc}`20-ordinal-regression`）为基础。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 用随机效用成分的联合分布表示选择概率；
- 由独立同分布的 Gumbel 误差推导多项 logit 模型；
- 解释尺度参数的作用以及无关选项独立性（IIA）；
- 描述多项 probit 模型，并说明其概率为何需要模拟计算；
- 用极大似然法估计选择模型，并用似然比指数评价拟合程度。
:::

## 符号说明

| 符号 | 含义 |
|---|---|
| $C$ | 全体选择集（总体的所有备选方案） |
| $J$ | $C$ 中备选方案的个数 |
| $C_n$, $J_n$ | 决策者 $n$ 的可行选择集及其大小，$J_n \le J$ |
| $U_{in}$ | 方案 $i$ 对决策者 $n$ 的效用 |
| $V_{in}$ | 系统（确定性）效用 |
| $\varepsilon_{in}$ | 随机效用成分 |
| $P_n(i)$ | 决策者 $n$ 选择方案 $i$ 的概率 |
| $N$ | 样本中决策者的人数 |

## 随机效用框架

### 选择集

记 $C$ 为包含总体所有潜在备选方案的**选择集**（choice set），$J$ 为方案个数。在交通方式研究中，$C$ 可能包含两种以上的方式，例如：

- 独自驾车；
- 合乘；
- 出租车；
- 摩托车；
- 自行车；
- 步行；
- 公交车；
- 轨道交通。

对某个出行者 $n$，其可行选择集 $C_n$ 可能小得多；例如，对没有汽车的人来说，独自驾车不可行。记 $J_n \le J$ 为可行方案的个数。

### 选择概率

方案 $i$ 对决策者 $n$ 的效用为 $U_{in} = V_{in} + \varepsilon_{in}$。在随机效用最大化下，

$$
\begin{aligned}
P_n(i) &= \Prob\bigl(U_{in} \ge U_{jn},\ \forall j \in C_n,\ j \ne i\bigr) \\
&= \Prob\bigl(V_{in} + \varepsilon_{in} \ge V_{jn} + \varepsilon_{jn},\ \forall j \in C_n,\ j \ne i\bigr) \\
&= \Prob\bigl(\varepsilon_{jn} \le V_{in} - V_{jn} + \varepsilon_{in},\ \forall j \in C_n,\ j \ne i\bigr).
\end{aligned}
$$ (eq-rum-probability)

等价地，

$$
P_n(i) = \Prob\Bigl[V_{in} + \varepsilon_{in} \ge \max_{j\in C_n,\ j\ne i}\bigl(V_{jn} + \varepsilon_{jn}\bigr)\Bigr].
$$

只要设定了**随机项的联合分布**，就可以由 {eq}`eq-rum-probability` 得到具体的多项选择模型。若随机项的联合密度为 $f(\varepsilon_{1n}, \ldots, \varepsilon_{J_nn})$，且不失一般性地设 $i$ 是 $C_n$ 中的第一个方案，则

$$
P_n(1) = \int_{\varepsilon_{1n}=-\infty}^{\infty}
\int_{\varepsilon_{2n}=-\infty}^{V_{1n}-V_{2n}+\varepsilon_{1n}} \cdots
\int_{\varepsilon_{J_nn}=-\infty}^{V_{1n}-V_{J_nn}+\varepsilon_{1n}}
f(\varepsilon_{1n}, \varepsilon_{2n}, \ldots, \varepsilon_{J_nn})\,
\mathrm{d}\varepsilon_{J_nn}\cdots\mathrm{d}\varepsilon_{2n}\,\mathrm{d}\varepsilon_{1n}.
$$

## 多项 logit 模型

### 假设

**多项 logit**（multinomial logit, MNL）模型假设随机项 $\varepsilon_{in}$：

- 在各方案之间相互独立；
- 同分布；
- 服从位置参数为 $\eta$、尺度参数为 $\mu > 0$ 的 Gumbel 分布，记为 $\mathrm{Gumbel}(\eta,\mu)$，其分布函数为 $F(\varepsilon) = \exp\{-e^{-\mu(\varepsilon-\eta)}\}$，均值为 $\eta + \gamma/\mu$，方差为 $\pi^2/(6\mu^2)$，其中 $\gamma \approx 0.57721$ 为 Euler–Mascheroni 常数。

Gumbel 分布假设可以看作对正态分布的近似，采用它主要是为了**解析上的方便**。然而，随机项独立同分布（i.i.d.）的假设是一个很强的限制：

- 它假定各效用随机成分的方差相等；
- 在某些情形下，随机项并不独立。

### 选择概率

:::{prf:theorem} 多项 logit 概率
:label: thm-mnl

若 $\varepsilon_{in}$（$i\in C_n$）独立同分布于 $\mathrm{Gumbel}(\eta,\mu)$，则

$$
P_n(i) = \frac{\exp(\mu V_{in})}{\sum_{j\in C_n}\exp(\mu V_{jn})},
\qquad i \in C_n .
$$ (eq-mnl)

尺度参数与随机项共同的标准差 $\sigma$ 的关系为

$$
\begin{gathered}
\sigma^2 = \frac{\pi^2}{6\mu^2} \quad\text{或} \\
\mu = \frac{\pi}{\sqrt{6}\,\sigma}.
\end{gathered}
$$
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

位置参数 $\eta$ 在差 $\varepsilon_{jn}-\varepsilon_{in}$ 中抵消，故不妨设 $\eta = 0$，即 $\varepsilon_{in} \overset{\text{i.i.d.}}{\sim} \mathrm{Gumbel}(0,\mu)$。将方案排序使 $i = 1$，并定义

$$
\begin{gathered}
U_n^{*} = \max_{j = 2,\ldots,J_n}\bigl(V_{jn} + \varepsilon_{jn}\bigr), \quad\text{于是} \\
P_n(1) = \Prob\bigl(V_{1n} + \varepsilon_{1n} \ge U_n^{*}\bigr).
\end{gathered}
$$

尺度参数相同的独立 Gumbel 变量的最大值仍服从同尺度的 Gumbel 分布：

$$
U_n^{*} \sim \mathrm{Gumbel}\!\left(\frac{1}{\mu}\ln\sum_{j=2}^{J_n}\exp(\mu V_{jn}),\ \mu\right).
$$

因此 $U_n^{*} = V_n^{*} + \varepsilon_n^{*}$，其中

$$
\begin{gathered}
V_n^{*} = \frac{1}{\mu}\ln\sum_{j=2}^{J_n}\exp(\mu V_{jn}) \quad\text{且} \\
\varepsilon_n^{*} \sim \mathrm{Gumbel}(0,\mu),
\end{gathered}
$$

$\varepsilon_n^{*}$ 与 $\varepsilon_{1n}$ 独立。于是

$$
\begin{aligned}
P_n(1) &= \Prob\bigl(V_{1n} + \varepsilon_{1n} \ge V_n^{*} + \varepsilon_n^{*}\bigr) \\
&= \Prob\bigl(\varepsilon_n^{*} - \varepsilon_{1n} \le V_{1n} - V_n^{*}\bigr).
\end{aligned}
$$

两个独立的 $\mathrm{Gumbel}(0,\mu)$ 变量之差服从 logistic 分布，其分布函数为 $F(x) = 1/(1+e^{-\mu x})$。因此

$$
\begin{aligned}
P_n(1) &= \frac{1}{1 + \exp\bigl[-\mu(V_{1n} - V_n^{*})\bigr]} \\
&= \frac{\exp(\mu V_{1n})}{\exp(\mu V_{1n}) + \exp(\mu V_n^{*})} \\
&= \frac{\exp(\mu V_{1n})}{\sum_{j\in C_n}\exp(\mu V_{jn})},
\end{aligned}
$$

因为 $\exp(\mu V_n^{*}) = \sum_{j=2}^{J_n}\exp(\mu V_{jn})$。
:::

尺度参数 $\mu$ 无法与 $V_{jn}$ 中的系数分开识别。对参数线性的效用 $V_{jn} = \beta_1x_{jn1} + \cdots + \beta_Kx_{jnK} = \bbeta\T\bx_{jn}$，模型只依赖于 $\mu V_{jn} = (\mu\bbeta)\T\bx_{jn}$。因此通常将 $\mu$ 标准化为 1：

$$
P_n(i) = \frac{\exp(V_{in})}{\sum_{j\in C_n}\exp(V_{jn})},
$$

并理解为估计出的系数已乘以了 $\mu$。只有两个方案时，$P_n(1) = 1/\{1 + \exp[-(V_{1n} - V_{2n})]\}$，这就是作用于效用差的二元 logistic 回归模型（{doc}`18-logistic-regression`）。

### 尺度参数的极端情形

两种极限情形说明了 $\mu = \pi/(\sqrt6\,\sigma)$ 的作用。

**情形 1：$\mu \to 0$，即 $\sigma \to +\infty$。**

$$
\lim_{\mu\to0}P_n(i) = \frac{1}{J_n}, \qquad \forall i\in C_n .
$$

当随机项的方差无限增大时，系统效用不再提供任何信息，所有可行方案被选中的可能性相同。

**情形 2：$\mu \to +\infty$，即 $\sigma \to 0$。**

$$
\begin{aligned}
\lim_{\mu\to+\infty}P_n(i)
&= \lim_{\mu\to+\infty}\frac{1}{1 + \sum_{j\in C_n,\ j\ne i}\exp\bigl[\mu(V_{jn}-V_{in})\bigr]} \\
&= \begin{cases}
1, & \text{若对所有 } j \ne i \text{ 有 } V_{in} > V_{jn}, \\
0, & \text{若存在某个 } j \text{ 使 } V_{in} < V_{jn} .
\end{cases}
\end{aligned}
$$

当随机项的方差趋于零时，模型成为**确定性选择模型**：关于偏好的全部信息都包含在系统效用中，$V_{in}$ 最高的方案必然被选中。

### 参数线性的 logit 模型

将 $V_{in}$ 限定为参数线性函数，得到

$$
P_n(i) = \frac{\exp\bigl(\mu\bbeta\T\bx_{in}\bigr)}{\sum_{j\in C_n}\exp\bigl(\mu\bbeta\T\bx_{jn}\bigr)},
$$

其中 $\bx_{in}$ 描述方案 $i$ 和决策者 $n$ 的属性。

### 无关选项独立性

:::{prf:property} 无关选项独立性（IIA）
:label: prop-iia

对给定的决策者，MNL 模型中任意两个方案 $i$ 与 $l$ 的选择概率之比与其他任何方案的系统效用无关：

$$
\begin{aligned}
\frac{P_n(i)}{P_n(l)} &= \frac{\exp(\mu V_{in})\big/\sum_{j\in C_n}\exp(\mu V_{jn})}{\exp(\mu V_{ln})\big/\sum_{j\in C_n}\exp(\mu V_{jn})} \\
&= \exp\bigl[\mu(V_{in}-V_{ln})\bigr].
\end{aligned}
$$
:::

IIA 通常被视为一个**缺点**：当某些方案彼此是近似替代品（**相关方案**）时，模型会失效。任何假设随机项相互独立且同分布的模型，在经典的**红巴士/蓝巴士问题**中都会给出违反直觉的概率：加入一辆与现有红巴士完全相同的蓝巴士，本应只分走巴士的份额，但在 IIA 下它会按比例从每个方案中分走份额。

## 多项 probit 模型

### 假设

**多项 probit**（multinomial probit, MNP）模型假设随机项服从联合正态分布：

$$
\bvarepsilon_n = (\varepsilon_{1n}, \varepsilon_{2n}, \ldots, \varepsilon_{J_nn})\T \sim \mathcal{N}(\bzero, \bSigma),
$$

对 $i\in C_n$ 有均值 $\E[\varepsilon_{in}] = 0$，协方差矩阵为一般的 $J_n\times J_n$ 矩阵

$$
\bSigma = \Bigl[\E\bigl[(\varepsilon_{in}-\E[\varepsilon_{in}])(\varepsilon_{jn}-\E[\varepsilon_{jn}])\bigr]\Bigr]_{J_n\times J_n},
$$

从而允许方案之间方差不等且相关。

### 选择概率

记 $\mathbf{u} = (u_1,\ldots,u_{J_n})\T$ 为效用向量 $\mathbf{U}_n = \mathbf{V}_n + \bvarepsilon_n$ 的一个取值，其中 $\mathbf{V}_n = (V_{1n},\ldots,V_{J_nn})\T$。则 $\mathbf{U}_n \sim \mathcal{N}(\mathbf{V}_n, \bSigma)$，且

$$
\begin{aligned}
P_n(i) &= \Prob\bigl(U_{in} \ge U_{jn},\ \forall j\ne i\bigr) \\
&= \int_{\{\mathbf{u}:\ u_i \ge u_j\ \forall j\ne i\}} (2\pi)^{-J_n/2}\,\lvert\bSigma\rvert^{-1/2} \exp\Bigl[-\tfrac12(\mathbf{u}-\mathbf{V}_n)\T\bSigma^{-1}(\mathbf{u}-\mathbf{V}_n)\Bigr] \,\mathrm{d}\mathbf{u}.
\end{aligned}
$$

- 选择概率没有闭式表达式。
- 在二元情形下，它退化为一元正态分布函数，容易计算。
- 一般情形下，需要用数值近似或**蒙特卡洛模拟**计算。

## 系统效用的设定

- 对于**不随方案变化**的变量（如决策者的收入），若在每个效用中以相同系数出现，则会在效用差中抵消。因此它只能以方案特定系数的形式进入至多 $J-1$ 个效用函数。
- 类似地，有 $L$ 个类别的分类变量至多用 $L-1$ 个虚拟变量编码。
- 通常不建议将连续变量分类化。

## 极大似然估计

### 似然函数

对含 $N$ 个决策者的样本，令

$$
y_{n,i} = \begin{cases} 1, & \text{若决策者 } n \text{ 选择方案 } i, \\ 0, & \text{其他.} \end{cases}
$$

系数 $\bbeta = (\beta_1,\ldots,\beta_K)\T$ 的似然函数为

$$
\mathcal{L}(\bbeta) = \prod_{n=1}^N\prod_{i\in C_n}P_n(i)^{\,y_{n,i}},
$$

对数似然函数为

$$
\begin{aligned}
\ell(\bbeta) &= \ln\mathcal{L}(\bbeta) \\
&= \sum_{n=1}^N\sum_{i\in C_n}y_{n,i}\ln P_n(i).
\end{aligned}
$$

### 估计

极大似然估计量 $\widehat{\bbeta}$ 是使观测到的选择最有可能出现的 $\bbeta$：

$$
\begin{aligned}
\widehat{\bbeta} &= \operatorname*{arg\,max}_{\bbeta}\mathcal{L}(\bbeta) \\
&= \operatorname*{arg\,max}_{\bbeta}\ell(\bbeta).
\end{aligned}
$$

对参数线性效用的 MNL 模型，$\ell$ 是凹函数，因此数值极大化是可靠的。

### 对数似然的性质与拟合优度

由于每个 $P_n(i) \le 1$，对数似然非正。

- $\ell = 0$ 对应 $\mathcal{L} = 1$：每个选择都以概率 1 被预测到（完美预测）。
- $\ell(\bzero)$ 是所有系数为零时的对数似然；此时所有可行方案的可能性相同，$\ell(\bzero) = -\sum_n \ln J_n$。

**似然比指数**（likelihood ratio index）衡量初始对数似然中被模型解释的比例：

$$
\rho^2 = 1 - \frac{\ell(\widehat{\bbeta})}{\ell(\bzero)}.
$$

- $\rho^2$ 的作用类似于线性回归中的 $R^2$。
- $0 \le \rho^2 \le 1$。
- 增加参数时 $\rho^2$ 不会减小。**调整**指数 $\bar\rho^2 = 1 - \bigl(\ell(\widehat{\bbeta}) - K\bigr)/\ell(\bzero)$ 对参数个数 $K$ 进行惩罚。

### 变量选择原则

决定是否保留某个变量时，应考虑：

- 估计系数的**符号和相对大小**；例如，价格的系数应为负；
- 对每个系数显著性的**统计检验**；
- **拟合优度**指标。

下表总结了一种常用的决策规则，区分了政策变量（分析所要评估的变量）和其他变量：

| 符号 | 显著性 | 政策变量 | 其他变量 |
|---|---|---|---|
| 正确 | 显著 | 保留 | 保留 |
| 正确 | 不显著 | 保留 | 可剔除 |
| 错误 | 显著 | 严重问题 | 剔除 |
| 错误 | 不显著 | 有问题 | 剔除 |

## 软件

- **Apollo**（R）：免费的程序包，可估计多种选择模型。它不依赖商业软件，可在任何能运行 R 的环境中使用。<http://www.apollochoicemodelling.com/manual.html>
- **mlogit**（R）：多项 logit 模型的估计；参见程序包说明文档 "Estimation of multinomial logit models in R: The mlogit packages"。<https://cran.r-project.org/package=mlogit>
- **Biogeme**（Python）：参数化选择模型的极大似然估计。<https://biogeme.epfl.ch/>

## 本章回顾
### 小结

- 一旦设定了随机项的联合分布，选择概率就由随机效用最大化得到。
- 独立同分布的 Gumbel 误差给出 MNL 模型 $P_n(i) = e^{\mu V_{in}}/\sum_{j}e^{\mu V_{jn}}$；尺度 $\mu$ 标准化为 1。
- MNL 隐含 IIA，在方案相关时失效；MNP 允许一般的协方差矩阵，但需要模拟计算。
- 选择模型用极大似然法估计；用 $\rho^2$ 和 $\bar\rho^2$ 衡量拟合程度。

### 常见错误

- 将不随方案变化的变量以相同系数放入每个效用中；它会被抵消，其系数不可识别。
- 不考虑尺度标准化 $\mu = 1$ 就比较不同数据集上的 MNL 系数。
- 在方案彼此近似可替代、IIA 不可信时仍使用 MNL。

## 参考文献

- Ben-Akiva, M., & Lerman, S. R. (1985). *Discrete choice analysis: Theory and application to travel demand*. MIT Press.
- Train, K. E. (2009). *Discrete choice methods with simulation* (2nd ed.). Cambridge University Press. <https://doi.org/10.1017/CBO9780511805271>
- Bierlaire, M. (n.d.). *Biogeme* [Computer software]. EPFL. <https://biogeme.epfl.ch/>
- Croissant, Y. (n.d.). *mlogit: Multinomial logit models* (R package) [Computer software]. <https://CRAN.R-project.org/package=mlogit>
- Hess, S., & Palma, D. (n.d.). *Apollo: Choice modelling in R* [Computer software]. <http://www.apollochoicemodelling.com/>

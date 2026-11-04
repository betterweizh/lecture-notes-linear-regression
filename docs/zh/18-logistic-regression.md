# logistic 回归

## 概述

许多响应是二元的：出行者是否选择公共交通，患者是否康复，贷款是否违约。**logistic 回归**（logistic regression，也称逻辑回归）把结果 $Y = 1$ 的概率建模为预测变量的函数。它是响应服从伯努利（或二项）分布、使用典则 logit 联系函数的 GLM（{doc}`17-generalized-linear-models`），也是最简单的**离散选择模型**：只有两个方案时，{doc}`21-multinomial-choice-models` 中的多项 logit 模型就退化为它。本章定义该模型，把系数解释为优势比，推导极大似然方程和 IRLS 算法，并介绍推断与模型评价。 本章以指数族、典则联系、偏差和 IRLS（{doc}`17-generalized-linear-models`）以及极大似然估计的大样本性质（{doc}`02-sampling-distributions-and-limit-theorems`）为基础。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 解释为什么对概率使用线性模型不合适，并写出 logistic 回归模型；
- 把系数解释为对数优势比，并计算边际效应；
- 推导对数似然函数、得分和 Hessian 矩阵，并描述 Newton–Raphson（IRLS）算法；
- 识别使极大似然估计不存在的分离现象；
- 进行 Wald 检验和似然比检验，并评价拟合程度；
- 把 logistic 回归与潜变量和随机效用表述联系起来。
:::

## 符号说明

| 符号 | 含义 |
|---|---|
| $Y_i \in \{0,1\}$ | 观测 $i$ 的二元响应 |
| $\pi_i = \Prob(Y_i = 1\mid\bx_i)$ | 成功概率 |
| $\eta_i = \bx_i\T\bbeta$ | 线性预测子（$\bx_i$ 的第一个元素为 1） |
| $\logit(\pi) = \ln\{\pi/(1-\pi)\}$ | 对数优势 |
| $\mathbf{W} = \diag\bigl(\pi_i(1-\pi_i)\bigr)$ | IRLS 权重矩阵 |

## 模型

### 为什么不用线性模型？

对概率使用线性模型 $\pi_i = \bx_i\T\bbeta$（即*线性概率模型*）可能给出 $[0,1]$ 之外的拟合值，而且方差 $\Var(Y_i) = \pi_i(1-\pi_i)$ 依赖于均值，违背了 {doc}`07-linear-regression-model` 中的同方差假设。logistic 回归则把线性预测子映射到 $(0,1)$ 内。

### logistic 回归

:::{prf:definition} logistic 回归模型
:label: def-logistic

响应 $Y_1,\ldots,Y_n$ 相互独立，$Y_i\mid\bx_i \sim \mathrm{Bernoulli}(\pi_i)$，且

$$
\begin{aligned}
\logit(\pi_i) &= \ln\frac{\pi_i}{1-\pi_i} \\
&= \bx_i\T\bbeta,
\end{aligned}
$$

等价地

$$
\begin{aligned}
\pi_i &= \frac{\exp(\bx_i\T\bbeta)}{1 + \exp(\bx_i\T\bbeta)} \\
&= \frac{1}{1 + \exp(-\bx_i\T\bbeta)} .
\end{aligned}
$$ (eq-logistic)
:::

函数 $\pi(\eta) = 1/(1+e^{-\eta})$ 是 logistic 分布函数（{doc}`01-probability-and-random-vectors`）；它呈 S 形、单调递增，关于 $\eta = 0$（此时 $\pi = 0.5$）对称。由 {prf:ref}`ex-bernoulli-canonical`，logit 是伯努利分布的典则联系函数，因此该模型是使用典则联系、方差函数为 $V(\pi) = \pi(1-\pi)$ 的 GLM。

### 潜变量与随机效用表述

设连续潜变量 $Y^{*} = \bx\T\bbeta + \varepsilon$ 只能通过 $Y = \mathbb{I}(Y^{*} > 0)$ 被观测到。若 $\varepsilon$ 服从标准 logistic 分布，则

$$
\begin{aligned}
\Prob(Y = 1\mid\bx)
&= \Prob(\varepsilon > -\bx\T\bbeta) \\
&= 1 - F(-\bx\T\bbeta) \\
&= F(\bx\T\bbeta),
\end{aligned}
$$

这就是 {eq}`eq-logistic`。若改为 $\varepsilon \sim \mathcal{N}(0,1)$，则得到 **probit 模型** $\Prob(Y=1\mid\bx) = \Phi(\bx\T\bbeta)$。在选择建模中，$Y^{*}$ 是两个方案的效用之差；若每个效用的误差独立服从 Gumbel 分布，其差服从 logistic 分布，二元 logit 模型就是多项 logit 模型在两个方案时的情形（{prf:ref}`thm-mnl`）。

## 系数的解释

### 优势与优势比

成功的**优势**（odds）为 $\pi/(1-\pi)$。在该模型下，对数优势关于预测变量是线性的，因此在其他预测变量不变时，$x_j$ 增加一个单位，对数优势改变 $\beta_j$：

$$
\begin{aligned}
\frac{\text{优势}(x_j + 1)}{\text{优势}(x_j)}
&= \frac{\exp(\beta_0 + \cdots + \beta_j(x_j+1) + \cdots)}{\exp(\beta_0 + \cdots + \beta_jx_j + \cdots)} \\
&= e^{\beta_j} .
\end{aligned}
$$

**优势比**（odds ratio）$e^{\beta_j}$ 与 $x_j$ 的取值无关：增加一个单位使优势乘以 $e^{\beta_j}$。它并不是把概率乘以 $e^{\beta_j}$；概率的变化取决于观测在 S 形曲线上的位置。

### 边际效应

对连续预测变量，概率的导数为

$$
\frac{\partial\pi}{\partial x_j} = \beta_j\,\pi(1-\pi),
$$

它在 $\pi = 0.5$ 处最大，等于 $\beta_j/4$。平均边际效应是对各观测的该量取平均。

## 估计

### 似然函数、得分与 Hessian 矩阵

对独立观测，似然函数为 $\prod_i\pi_i^{y_i}(1-\pi_i)^{1-y_i}$，记 $\eta_i = \bx_i\T\bbeta$，对数似然函数为

$$
\begin{aligned}
\ell(\bbeta)
&= \sum_{i=1}^n\bigl[y_i\ln\pi_i + (1-y_i)\ln(1-\pi_i)\bigr] \\
&= \sum_{i=1}^n\bigl[y_i\,\eta_i - \ln\bigl(1 + e^{\eta_i}\bigr)\bigr].
\end{aligned}
$$ (eq-logistic-loglik)

利用 $\partial\ln(1+e^{\eta})/\partial\eta = \pi$ 和 $\partial\pi/\partial\eta = \pi(1-\pi)$，

$$
\begin{aligned}
\nabla\ell(\bbeta) &= \sum_{i=1}^n (y_i - \pi_i)\,\bx_i \\
&= \bX\T(\by - \boldsymbol\pi), \\
\frac{\partial^2\ell}{\partial\bbeta\,\partial\bbeta\T} &= -\sum_{i=1}^n \pi_i(1-\pi_i)\,\bx_i\bx_i\T \\
&= -\bX\T\mathbf{W}\bX ,
\end{aligned}
$$

其中 $\boldsymbol\pi = (\pi_1,\ldots,\pi_n)\T$。当 $\bX$ 列满秩时 Hessian 矩阵负定，因此对数似然严格凹（{prf:ref}`thm-optimality`），极大似然估计（若存在）唯一。似然方程 $\bX\T(\by - \widehat{\boldsymbol\pi}) = \bzero$ 没有闭式解。

### Newton–Raphson 与 IRLS

由于 logit 联系是典则联系，Hessian 矩阵不依赖于 $\by$，Newton–Raphson 法与 Fisher 得分法一致：

$$
\begin{aligned}
\bbeta^{(t+1)}
&= \bbeta^{(t)} + (\bX\T\mathbf{W}\bX)^{-1}\bX\T(\by - \boldsymbol\pi) \\
&= (\bX\T\mathbf{W}\bX)^{-1}\bX\T\mathbf{W}\mathbf{z},
\qquad
z_i = \eta_i + \frac{y_i - \pi_i}{\pi_i(1-\pi_i)},
\end{aligned}
$$

其中 $\mathbf{W}$、$\boldsymbol\pi$ 和 $\boldsymbol\eta$ 都在 $\bbeta^{(t)}$ 处取值。每一步都是工作响应 $\mathbf{z}$ 对 $\bX$ 的加权最小二乘回归：这就是 {doc}`17-generalized-linear-models` 中 $g'(\pi) = 1/\{\pi(1-\pi)\}$、$V(\pi) = \pi(1-\pi)$ 时的 IRLS 算法。收敛时，$\widehat{\Cov}(\widehat{\bbeta}) = (\bX\T\widehat{\mathbf{W}}\bX)^{-1}$，即 Fisher 信息的逆。

### 分离

若预测变量的某个线性组合能把观测完全分开，即存在 $\mathbf{b}$，使得当 $y_i = 1$ 时 $\bx_i\T\mathbf{b} > 0$，当 $y_i = 0$ 时 $\bx_i\T\mathbf{b} < 0$，那么沿 $\bbeta = c\,\mathbf{b}$、$c\to\infty$ 方向对数似然趋于 $0$ 而单调增加，**极大似然估计不存在**。软件此时会报告极大的系数和标准误，或给出不收敛警告。处理方法包括删除或合并造成分离的预测变量、收集更多数据，或使用惩罚（岭或 Firth）logistic 回归。

### 分组二项数据

若在每个不同的预测变量取值处有 $m_i$ 次试验、$y_i$ 次成功，则 $Y_i \sim \mathrm{Binomial}(m_i, \pi_i)$，对数似然变为 $\sum_i\bigl[y_i\eta_i - m_i\ln(1+e^{\eta_i})\bigr]$ 加上一个常数。其估计与把数据展开为单个伯努利观测所得的估计相同。

## 推断

由 {prf:ref}`thm-mle-asymptotic`，可以使用下列大样本方法。

- $H_0:\beta_j = 0$ 的 **Wald 检验**：$Z = \widehat{\beta}_j/\mathrm{SE}(\widehat{\beta}_j)$ 在 $H_0$ 下近似服从 $\mathcal{N}(0,1)$，其中 $\mathrm{SE}(\widehat{\beta}_j)$ 是 $(\bX\T\widehat{\mathbf{W}}\bX)^{-1}$ 第 $j$ 个对角元的平方根。
- **优势比的置信区间**：对 Wald 区间取指数，$\exp\bigl(\widehat{\beta}_j \pm z_{1-\alpha/2}\,\mathrm{SE}(\widehat{\beta}_j)\bigr)$。
- 嵌套模型的**似然比检验**：$G^2 = -2\bigl[\ell(\widehat{\bbeta}_{\text{reduced}}) - \ell(\widehat{\bbeta}_{\text{full}})\bigr]$ 在 $H_0$ 下近似服从 $\chi^2(d)$，$d$ 为删去的系数个数。它就是 {doc}`19-poisson-and-count-regression` 中的偏差下降检验在二元数据上的应用，通常比 Wald 检验更可靠。

## 模型评价

### 偏差

logistic 回归的偏差为

$$
D = 2\sum_{i=1}^n\left[y_i\ln\frac{y_i}{\widehat{\pi}_i} + (1-y_i)\ln\frac{1-y_i}{1-\widehat{\pi}_i}\right],
$$

约定 $0\ln 0 = 0$。对未分组的二元数据，饱和模型的对数似然为零，故 $D = -2\ell(\widehat{\bbeta})$；此时 $D$ **不**近似服从 $\chi^2(n-p)$，不应作为绝对的拟合优度检验，但偏差之差仍然有效。对 $m_i$ 较大的分组数据，$D \approx \chi^2(\text{组数} - p)$。

### 其他指标

- **伪 $R^2$。** McFadden 伪 $R^2$ 为 $R^2_{\mathrm{McF}} = 1 - \ell(\widehat{\bbeta})/\ell(\widehat{\beta}_0)$，把模型与只含截距的模型相比较；它就是 {doc}`21-multinomial-choice-models` 中以不同基准计算的似然比指数 $\rho^2$。
- **Hosmer–Lemeshow 检验。** 按拟合概率把观测分组（通常分为十组），用 $\chi^2$ 统计量比较观测频数与期望频数。
- **分类。** 当 $\widehat{\pi} > c$（阈值 $c$，常取 $0.5$）时预测 $\widehat{Y} = 1$，由此得到混淆矩阵、灵敏度和特异度；ROC 曲线及其曲线下面积（AUC）概括了所有阈值下的表现。
- **残差。** Pearson 残差 $(y_i - \widehat{\pi}_i)/\sqrt{\widehat{\pi}_i(1-\widehat{\pi}_i)}$ 和偏差残差的定义与 {doc}`19-poisson-and-count-regression` 中相同。

## 本章回顾
### 小结

- logistic 回归对伯努利响应建模 $\logit(\pi_i) = \bx_i\T\bbeta$；它是二元数据的典则联系 GLM，也是多项 logit 模型的二元情形。
- $e^{\beta_j}$ 是优势比；$\partial\pi/\partial x_j = \beta_j\pi(1-\pi)$。
- 得分为 $\bX\T(\by-\boldsymbol\pi)$，Hessian 矩阵为 $-\bX\T\mathbf{W}\bX$；对数似然严格凹，可用 IRLS 求极大。
- 存在分离时 MLE 不存在。
- 使用 Wald 检验和似然比检验；用伪 $R^2$、Hosmer–Lemeshow 检验和分类指标评价拟合。

### 常见错误

- 把 $e^{\beta_j}$ 理解为概率之比（相对风险）；它是优势之比，只有当概率很小时才接近相对风险。
- 把未分组二元数据的残差偏差当作 $\chi^2$ 拟合优度检验。
- 忽略由分离引起的不收敛警告。
- 直接比较 logit 与 probit 的系数；二者尺度不同（logit 系数约为 probit 系数的 $1.6$–$1.8$ 倍）。

## 参考文献

- Agresti, A. (2013). *Categorical data analysis* (3rd ed.). Wiley.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied logistic regression* (3rd ed.). Wiley. <https://doi.org/10.1002/9781118548387>
- McCullagh, P., & Nelder, J. A. (1989). *Generalized linear models* (2nd ed.). Chapman & Hall.
- Train, K. E. (2009). *Discrete choice methods with simulation* (2nd ed.). Cambridge University Press. <https://doi.org/10.1017/CBO9780511805271>

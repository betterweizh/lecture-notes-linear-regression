# 广义线性模型

## 概述

正态线性模型假设响应变量连续、服从正态分布，且其均值是预测变量的线性函数。许多响应变量并非如此：计数是非负整数，二元结果只取 $0$ 和 $1$，而且它们的方差依赖于均值。**广义线性模型**（generalized linear model, GLM）保留线性预测子，但允许响应变量服从指数族中的任意分布，并通过联系函数将均值与线性预测子联系起来。本章建立第四部分其余各章（二元、计数、有序和多项模型）所用的框架。 本章以极大似然（{doc}`08-ols-and-ml-estimation`）、极大似然估计的大样本性质（{doc}`02-sampling-distributions-and-limit-theorems`）以及梯度、Hessian 矩阵和泰勒展开（{doc}`06-matrix-calculus`）为基础。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 识别 GLM 的随机成分、线性预测子和联系函数；
- 将常见分布写成典则指数族形式，并由累积量函数 $b(\theta)$ 推导其均值和方差；
- 定义典则联系函数并说明其优点；
- 写出 GLM 的对数似然函数并定义偏差；
- 描述 Newton–Raphson 法、Fisher 得分法和迭代重加权最小二乘（IRLS）算法。
:::

## 符号说明

| 符号 | 含义 |
|---|---|
| $Y_i$, $y_i$ | 随机响应及其观测值，$i = 1,\ldots,n$ |
| $\bx_i \in \R^p$ | 观测 $i$ 的预测变量向量（含截距时第一个元素为 $1$） |
| $\mu_i = \E[Y_i\mid\bx_i]$ | 条件均值 |
| $\eta_i = \bx_i\T\bbeta$ | 线性预测子 |
| $g$ | 联系函数，$g(\mu_i) = \eta_i$ |
| $\theta_i$ | 典则（自然）参数 |
| $\phi$ | 离散参数 |
| $b(\theta)$, $c(y,\phi)$ | 确定分布的已知函数 |
| $V(\mu)$ | 方差函数，$\Var(Y) = \phi\,V(\mu)$ |
| $\ell(\bbeta)$ | 对数似然函数 |

## 从线性模型到 GLM

正态线性模型有两个成分：

- **随机成分。** 给定 $\bx$ 时，响应 $Y$ 连续且服从正态分布，均值为 $\mu(\bx) = \E[Y\mid\bx]$。
- **系统成分。** 均值关于预测变量线性：$\mu(\bx) = \bx\T\bbeta$。

GLM 对两个成分都做了推广：

- **随机成分。** $Y\mid\bx$ 服从指数族中的某个分布。
- **系统成分与联系函数。** 联系函数 $g$ 将均值与线性预测子联系起来：

  $$
  \begin{aligned}
  g\bigl(\mu(\bx)\bigr) &= \eta \\
  &= \bx\T\bbeta .
  \end{aligned}
  $$

正态线性模型是响应服从正态分布、使用恒等联系 $g(\mu) = \mu$ 的特例。

## 指数族

### 一般定义

:::{prf:definition} 指数族
:label: def-exp-family

设分布族 $\{P_{\boldsymbol\vartheta} : \boldsymbol\vartheta \in \Theta\}$，$\Theta \subseteq \R^m$。若存在 $\boldsymbol\vartheta$ 的实值函数 $\eta_1,\ldots,\eta_m$ 和 $B$，以及 $x\in\R^q$ 的实值函数 $T_1,\ldots,T_m$ 和 $h$，使 $P_{\boldsymbol\vartheta}$ 的密度（概率质量函数或概率密度函数）可以写成

$$
p_{\boldsymbol\vartheta}(x)
= \exp\Bigl[\sum_{j=1}^m \eta_j(\boldsymbol\vartheta)\,T_j(x) - B(\boldsymbol\vartheta)\Bigr]\,h(x),
$$

则称该分布族为 $\R^q$ 上的 $m$ 参数**指数族**（exponential family）。
:::

仅在此定义中，$\eta_j(\cdot)$ 表示分布族的自然参数函数，而非线性预测子。

:::{prf:example} 正态分布
:label: ex-normal-exp-family

设 $X\sim\mathcal{N}(\mu,\sigma^2)$，$\boldsymbol\vartheta = (\mu,\sigma^2)$。其密度为

$$
p_{\boldsymbol\vartheta}(x)
= \exp\!\left(\frac{\mu}{\sigma^2}x - \frac{1}{2\sigma^2}x^2 - \frac{\mu^2}{2\sigma^2}\right)\frac{1}{\sigma\sqrt{2\pi}},
$$

这是一个二参数指数族，其中

$$
\begin{gathered}
\eta_1 = \frac{\mu}{\sigma^2}, \qquad  \eta_2 = -\frac{1}{2\sigma^2}, \\
T_1(x) = x, \qquad  T_2(x) = x^2, \\
B(\boldsymbol\vartheta) = \frac{\mu^2}{2\sigma^2} + \ln\bigl(\sigma\sqrt{2\pi}\bigr), \qquad  h(x) = 1 .
\end{gathered}
$$

若 $\sigma^2$ 已知，它成为 $\R$ 上的单参数指数族：

$$
\begin{gathered}
\eta = \frac{\mu}{\sigma^2}, \\
T(x) = x, \\
B(\mu) = \frac{\mu^2}{2\sigma^2}, \\
h(x) = \frac{e^{-x^2/(2\sigma^2)}}{\sigma\sqrt{2\pi}} .
\end{gathered}
$$
:::

指数族的其他成员包括：

- 离散分布：伯努利分布、二项分布（试验次数已知）、泊松分布、负二项分布（形状参数已知）；
- 连续分布：伽马分布、逆伽马分布、逆高斯分布、卡方分布和贝塔分布。

### 典则单参数形式

GLM 对标量响应 $y\in\R$ 使用如下单参数形式：

$$
f_\theta(y) = \exp\left(\frac{y\theta - b(\theta)}{\phi} + c(y,\phi)\right)
$$ (eq-canonical-exp-family)

其中 $b(\cdot)$ 和 $c(\cdot,\cdot)$ 是已知函数。

- 若 $\phi$ 已知，这是以 $\theta$ 为**典则参数**（canonical parameter）的单参数指数族。
- 若 $\phi$ 未知，它可能是也可能不是二参数指数族。$\phi$ 称为**离散参数**（dispersion parameter）。
- 本章假设 $\phi$ 已知。

:::{prf:example} 方差已知的正态分布
:label: ex-normal-canonical

$$
\begin{aligned}
f_\theta(y)
&= \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right) \\
&= \exp\left\{\frac{y\mu - \tfrac12\mu^2}{\sigma^2}
   - \frac12\left(\frac{y^2}{\sigma^2} + \ln(2\pi\sigma^2)\right)\right\}.
\end{aligned}
$$

因此 $\theta = \mu$，$\phi = \sigma^2$，$b(\theta) = \theta^2/2$，$c(y,\phi) = -\tfrac12\bigl(y^2/\phi + \ln(2\pi\phi)\bigr)$。
:::

:::{prf:example} 泊松分布
:label: ex-poisson-canonical

$$
\begin{aligned}
f(y\mid\mu) &= \frac{\mu^y}{y!}e^{-\mu} \\
&= \exp\bigl[y\ln\mu - \mu - \ln(y!)\bigr], \qquad y = 0,1,2,\ldots
\end{aligned}
$$

因此 $\theta = \ln\mu$，$b(\theta) = e^\theta$，$\phi = 1$，$c(y,\phi) = -\ln(y!)$。
:::

:::{prf:example} 伯努利分布
:label: ex-bernoulli-canonical

对 $y\in\{0,1\}$，$\Prob(Y=1) = \mu$，

$$
\begin{aligned}
f(y\mid\mu) &= \mu^y(1-\mu)^{1-y} \\
&= \exp\left[y\ln\frac{\mu}{1-\mu} + \ln(1-\mu)\right].
\end{aligned}
$$

因此 $\theta = \ln\{\mu/(1-\mu)\}$，于是 $\mu = e^\theta/(1+e^\theta)$，$b(\theta) = -\ln(1-\mu) = \ln(1+e^\theta)$，且 $\phi = 1$，$c(y,\phi) = 0$。
:::

| | 正态 | 泊松 | 伯努利 |
|---|---|---|---|
| 记号 | $\mathcal{N}(\mu,\sigma^2)$ | $\mathrm{Poisson}(\mu)$ | $\mathrm{Bernoulli}(\mu)$ |
| $y$ 的取值范围 | $(-\infty,\infty)$ | $\{0,1,2,\ldots\}$ | $\{0,1\}$ |
| $\phi$ | $\sigma^2$ | $1$ | $1$ |
| $b(\theta)$ | $\theta^2/2$ | $e^\theta$ | $\ln(1+e^\theta)$ |
| $c(y,\phi)$ | $-\tfrac12\bigl(y^2/\phi + \ln(2\pi\phi)\bigr)$ | $-\ln(y!)$ | $0$ |

## 均值与方差

记 $\ell(\theta) = \ln f_\theta(Y)$ 为单个观测的对数似然。对恒等式 $\int f_\theta(y)\,dy \equiv 1$（离散情形为求和）关于 $\theta$ 求一阶和二阶导数，得到两个恒等式：

$$
\begin{gathered}
\E\!\left[\frac{\partial\ell}{\partial\theta}\right] = 0, \\
\E\!\left[\frac{\partial^2\ell}{\partial\theta^2}\right] + \E\!\left[\left(\frac{\partial\ell}{\partial\theta}\right)^2\right] = 0 .
\end{gathered}
$$ (eq-score-identities)

### 期望

由 {eq}`eq-canonical-exp-family`，$\ell(\theta) = \{Y\theta - b(\theta)\}/\phi + c(Y,\phi)$，因此

$$
\frac{\partial\ell}{\partial\theta} = \frac{Y - b'(\theta)}{\phi}.
$$

由第一个恒等式得 $0 = \{\E[Y] - b'(\theta)\}/\phi$，从而

$$
\begin{aligned}
\E[Y] &= \mu \\
&= b'(\theta).
\end{aligned}
$$ (eq-glm-mean)

### 方差

二阶导数为 $\partial^2\ell/\partial\theta^2 = -b''(\theta)/\phi$，而由 {eq}`eq-glm-mean`，$\partial\ell/\partial\theta = (Y-\E[Y])/\phi$。于是第二个恒等式变为

$$
0 = -\frac{b''(\theta)}{\phi} + \frac{\Var(Y)}{\phi^2},
$$

从而

$$
\Var(Y) = \phi\,b''(\theta).
$$ (eq-glm-variance)

由于 $\mu = b'(\theta)$，方差可以写成均值的函数 $\Var(Y) = \phi\,V(\mu)$，其中 $V(\mu) = b''\bigl((b')^{-1}(\mu)\bigr)$ 称为**方差函数**（variance function）。例如，正态分布 $V(\mu) = 1$，泊松分布 $V(\mu) = \mu$，伯努利分布 $V(\mu) = \mu(1-\mu)$。

## 联系函数

### 定义

我们关心的参数是 $\bbeta$，它必须出现在似然函数中才能用极大似然法估计。**联系函数**（link function）$g$ 将线性预测子 $\eta = \bx\T\bbeta$ 与均值 $\mu$ 联系起来：

$$
\begin{gathered}
g(\mu) = \bx\T\bbeta, \\
\mu = g^{-1}\bigl(\bx\T\bbeta\bigr).
\end{gathered}
$$

要求联系函数单调且可微，从而可逆。

常用联系函数：

| 联系 | $g(\mu)$ | 典型用途 |
|---|---|---|
| 恒等 | $\mu$ | 正态响应 |
| 对数 | $\ln\mu$ | 计数、正值响应 |
| logit | $\ln\{\mu/(1-\mu)\}$ | 二元响应 |
| probit | $\Phi^{-1}(\mu)$，$\Phi$ 为标准正态分布函数 | 二元响应 |
| 互补双对数 | $\ln\{-\ln(1-\mu)\}$ | 二元响应 |
| 倒数 | $1/\mu$ | 伽马响应 |

### 典则联系函数

:::{prf:definition} 典则联系函数
:label: def-canonical-link

将均值映射为典则参数的联系函数 $g(\mu) = \theta$ 称为**典则联系函数**（canonical link）。由于 $\mu = b'(\theta)$，它为

$$
g(\mu) = (b')^{-1}(\mu).
$$
:::

若 $\phi > 0$，典则联系函数严格递增：由 {eq}`eq-glm-variance`，对任何非退化分布有 $b''(\theta) = \Var(Y)/\phi > 0$，所以 $b'$ 严格递增，其反函数也严格递增。

对伯努利分布，$b'(\theta) = e^\theta/(1+e^\theta) = \mu$，解出 $\theta$ 即得典则联系函数 $g(\mu) = \ln\{\mu/(1-\mu)\} = \logit(\mu)$，由此得到 logistic 回归（{doc}`18-logistic-regression`）。其他例子：

| 分布 | $b(\theta)$ | 典则联系 $g(\mu)$ |
|---|---|---|
| 正态 | $\theta^2/2$ | $\mu$ |
| 泊松 | $e^\theta$ | $\ln\mu$ |
| 伯努利 | $\ln(1+e^\theta)$ | $\ln\{\mu/(1-\mu)\}$ |
| 伽马 | $-\ln(-\theta)$ | $-1/\mu$ |

## GLM 及其似然函数

### 模型

设 $(\bx_i, Y_i) \in \R^p\times\R$，$i = 1,\ldots,n$，为相互独立的数据对，给定 $\bx_i$ 时 $Y_i$ 的条件分布密度属于典则指数族：

$$
f_{\theta_i}(y_i) = \exp\left[\frac{y_i\theta_i - b(\theta_i)}{\phi} + c(y_i,\phi)\right],
$$

响应向量为 $\bY = (Y_1,\ldots,Y_n)\T$，设计矩阵 $\bX\in\R^{n\times p}$ 的第 $i$ 行为 $\bx_i\T$。均值 $\mu_i = b'(\theta_i)$ 通过联系函数依赖于预测变量：

$$
\begin{aligned}
g(\mu_i) &= \eta_i \\
&= \bx_i\T\bbeta .
\end{aligned}
$$

给定联系函数 $g$，典则参数是 $\bbeta$ 的函数：

$$
\begin{aligned}
\theta_i &= (b')^{-1}(\mu_i) \\
&= (b')^{-1}\bigl(g^{-1}(\bx_i\T\bbeta)\bigr) \equiv h(\bx_i\T\bbeta), \\
h &= (b')^{-1}\circ g^{-1} \\
&= (g\circ b')^{-1}.
\end{aligned}
$$

若 $g$ 为典则联系函数，则 $h$ 为恒等函数，$\theta_i = \eta_i$。

### 对数似然

略去与 $\bbeta$ 无关的项，对数似然为

$$
\begin{aligned}
\ell(\bbeta) &= \sum_{i=1}^n \frac{Y_i\theta_i - b(\theta_i)}{\phi} \\
&= \sum_{i=1}^n \frac{Y_i\,h(\bx_i\T\bbeta) - b\bigl(h(\bx_i\T\bbeta)\bigr)}{\phi}.
\end{aligned}
$$ (eq-glm-loglik)

使用典则联系函数时，它简化为

$$
\ell(\bbeta) = \sum_{i=1}^n \frac{Y_i\,\bx_i\T\bbeta - b(\bx_i\T\bbeta)}{\phi}.
$$

其 Hessian 矩阵为 $-\phi^{-1}\sum_i b''(\bx_i\T\bbeta)\,\bx_i\bx_i\T$，当 $\phi > 0$ 且 $\bX$ 列满秩时负定。因此，使用典则联系函数时对数似然**严格凹**，极大似然估计量（若存在）**唯一**。使用其他联系函数时，对数似然不一定是凹的，可能有多个局部极大值点。

### 偏差

**饱和模型**（saturated model）让每个观测拥有自己的均值 $\widehat{\mu}_i = y_i$，达到可能的最大对数似然 $\ell_{\mathrm{sat}}$。拟合模型的**偏差**（deviance）将其对数似然 $\ell(\widehat{\bbeta})$ 与饱和模型相比较：

$$
D = 2\phi\bigl[\ell_{\mathrm{sat}} - \ell(\widehat{\bbeta})\bigr].
$$ (eq-deviance)

对正态模型，$D = \mathit{SS}_E$。对 $\phi = 1$ 的模型（如泊松模型），两个嵌套模型的偏差之差等于 $-2\ell(\widehat{\bbeta})$ 之差，即 {doc}`19-poisson-and-count-regression` 中使用的似然比统计量。

## 估计算法

似然方程 $\partial\ell/\partial\bbeta = \bzero$ 通常没有闭式解，需要迭代求解。常用方法有：

- Newton–Raphson 法；
- Fisher 得分法（Fisher scoring）；
- 迭代重加权最小二乘（IRLS）。

### Newton–Raphson 法

记 $\nabla\ell(\bbeta)$ 为对数似然的梯度（得分），$\mathbf{H}_\ell(\bbeta)$ 为其 Hessian 矩阵。Newton–Raphson 法的迭代为

$$
\bbeta^{(t+1)} = \bbeta^{(t)} - \mathbf{H}_\ell\bigl(\bbeta^{(t)}\bigr)^{-1}\nabla\ell\bigl(\bbeta^{(t)}\bigr),
$$

每一步极大化 $\ell$ 的二次近似。该方法是纯数值方法，并不利用数据是随机的这一事实。

### Fisher 得分法

计算 Hessian 矩阵可能很复杂。Fisher 得分法用其期望代替：**Fisher 信息**

$$
\mathbf{I}(\bbeta) = -\E\bigl[\mathbf{H}_\ell(\bbeta)\bigr]
$$

是正定的，可以在 Newton–Raphson 迭代中代替 $-\mathbf{H}_\ell$：

$$
\bbeta^{(t+1)} = \bbeta^{(t)} + \mathbf{I}\bigl(\bbeta^{(t)}\bigr)^{-1}\nabla\ell\bigl(\bbeta^{(t)}\bigr).
$$

Fisher 得分法与 Newton–Raphson 法的收敛性质基本相同，但 $\mathbf{I}$ 往往比 $\mathbf{H}_\ell$ 更容易计算。使用典则联系函数时，Hessian 矩阵不依赖于 $\bY$，两种方法完全一致。

### 迭代重加权最小二乘

对 GLM，Fisher 得分法的迭代可以写成一个加权最小二乘问题。给定当前估计，构造**工作响应**和**权重**

$$
\begin{gathered}
z_i = \eta_i + (y_i - \mu_i)\,g'(\mu_i), \\
w_i = \frac{1}{V(\mu_i)\,\{g'(\mu_i)\}^2},
\end{gathered}
$$

它们在 $\bbeta^{(t)}$ 处取值，并令 $\mathbf{W} = \diag(w_1,\ldots,w_n)$。于是

$$
\bbeta^{(t+1)} = (\bX\T\mathbf{W}\bX)^{-1}\bX\T\mathbf{W}\mathbf{z}.
$$

每次迭代都是 $\mathbf{z}$ 对 $\bX$ 的一次加权线性回归，因此该算法称为 IRLS。收敛时，$\phi\,(\bX\T\mathbf{W}\bX)^{-1}$ 是 $\widehat{\bbeta}$ 协方差矩阵的估计。

## 本章回顾
### 小结

- GLM 由指数族响应、线性预测子 $\eta = \bx\T\bbeta$ 和联系函数 $g(\mu) = \eta$ 组成。
- 在典则形式下，$\E[Y] = b'(\theta)$，$\Var(Y) = \phi\,b''(\theta)$。
- 典则联系函数 $g = (b')^{-1}$ 使对数似然严格凹。
- 偏差将模型与饱和模型相比较；偏差之差用于比较嵌套模型。
- GLM 用 Newton–Raphson 法、Fisher 得分法或与之等价的 IRLS 算法拟合。

### 常见错误

- 把联系函数与对响应变量的变换混为一谈。使用对数联系的 GLM 建模的是 $\ln\E[Y]$，而不是 $\E[\ln Y]$。
- 认为任何联系函数都使对数似然为凹函数；只有典则联系函数能保证这一点。
- 把偏差当作 $-2\ell(\widehat{\bbeta})$；偏差是相对于饱和模型度量的。

## 参考文献

- McCullagh, P., & Nelder, J. A. (1989). *Generalized linear models* (2nd ed.). Chapman & Hall.
- Rigollet, P. (2016). *18.650 Statistics for applications* [Lecture notes]. MIT OpenCourseWare. <https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/>
- Wikipedia contributors. (n.d.). Generalized linear model. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Generalized_linear_model>

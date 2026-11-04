# 向量与矩阵求导

## 概述

最小二乘、极大似然和惩罚回归都是对系数向量的标量函数求极小或极大。令其对整个向量的导数为零，就能一步得到估计方程。本章约定布局规范，列出书中用到的求导法则，并介绍二阶条件、凸性以及 lasso 所需的次梯度。 本章以 {doc}`05-linear-algebra` 和一元微积分为基础。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 计算向量的线性函数和二次函数的梯度；
- 对最小二乘目标函数求导并推出正规方程；
- 用 Hessian 矩阵判断凸性并陈述二阶条件；
- 写出二阶泰勒展开，即牛顿法的基础；
- 计算绝对值函数的次微分。
:::

## 布局规范

设 $f:\R^p\to\R$ 是向量 $\bbeta = (\beta_1,\ldots,\beta_p)\T$ 的标量函数。本书采用**分母布局**（梯度布局）：标量对列向量的导数是同维数的列向量，

$$
\begin{aligned}
\frac{\partial f}{\partial\bbeta} &= \nabla f(\bbeta) \\
&= \left(\frac{\partial f}{\partial\beta_1},\ \ldots,\ \frac{\partial f}{\partial\beta_p}\right)\T,
\end{aligned}
$$

二阶导数矩阵为 $p\times p$ 的 **Hessian 矩阵**

$$
\frac{\partial^2 f}{\partial\bbeta\,\partial\bbeta\T}
= \left[\frac{\partial^2 f}{\partial\beta_j\,\partial\beta_l}\right]_{j,l=1}^{p},
$$

当二阶导数连续时它是对称的。

## 基本法则

设 $\mathbf{a}\in\R^p$ 和 $\mathbf{A}\in\R^{p\times p}$ 为常数。

| 函数 $f(\bbeta)$ | 梯度 $\partial f/\partial\bbeta$ | Hessian 矩阵 |
|---|---|---|
| $\mathbf{a}\T\bbeta = \bbeta\T\mathbf{a}$ | $\mathbf{a}$ | $\mathbf{0}$ |
| $\bbeta\T\bbeta = \lVert\bbeta\rVert_2^2$ | $2\bbeta$ | $2\bI_p$ |
| $\bbeta\T\mathbf{A}\bbeta$ | $(\mathbf{A} + \mathbf{A}\T)\bbeta$ | $\mathbf{A} + \mathbf{A}\T$ |
| $\bbeta\T\mathbf{A}\bbeta$，$\mathbf{A}$ 对称 | $2\mathbf{A}\bbeta$ | $2\mathbf{A}$ |

对向量值函数 $\mathbf{g}:\R^p\to\R^m$，**Jacobian 矩阵**是元素为 $\partial g_i/\partial\beta_j$ 的 $m\times p$ 矩阵 $\partial\mathbf{g}/\partial\bbeta\T$。例如 $\partial(\bX\bbeta)/\partial\bbeta\T = \bX$。标量复合函数 $f(\mathbf{g}(\bbeta))$ 的**链式法则**为

$$
\frac{\partial f}{\partial\bbeta}
= \left(\frac{\partial\mathbf{g}}{\partial\bbeta\T}\right)\T\frac{\partial f}{\partial\mathbf{g}} .
$$

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

对二次型，写成 $\bbeta\T\mathbf{A}\bbeta = \sum_{j}\sum_{l}a_{jl}\beta_j\beta_l$。对 $\beta_m$ 求导得

$$
\begin{aligned}
\frac{\partial}{\partial\beta_m}\sum_{j,l}a_{jl}\beta_j\beta_l &= \sum_{l}a_{ml}\beta_l + \sum_{j}a_{jm}\beta_j \\
&= (\mathbf{A}\bbeta)_m + (\mathbf{A}\T\bbeta)_m .
\end{aligned}
$$

其余法则是其特例或可类似得到。
:::

## 最小二乘目标函数

:::{prf:example} 残差平方和的梯度
:label: ex-gradient-rss

令 $S(\bbeta) = \lVert\by - \bX\bbeta\rVert_2^2$。展开得

$$
\begin{aligned}
S(\bbeta)
&= (\by - \bX\bbeta)\T(\by - \bX\bbeta) \\
&= \by\T\by - 2\by\T\bX\bbeta + \bbeta\T\bX\T\bX\bbeta .
\end{aligned}
$$

由上表，取 $\mathbf{a} = \bX\T\by$ 和对称矩阵 $\mathbf{A} = \bX\T\bX$，

$$
\begin{aligned}
\frac{\partial S}{\partial\bbeta} &= -2\bX\T\by + 2\bX\T\bX\bbeta \\
&= -2\bX\T(\by - \bX\bbeta), \\
\frac{\partial^2 S}{\partial\bbeta\,\partial\bbeta\T} &= 2\bX\T\bX .
\end{aligned}
$$

用链式法则（$\mathbf{g}(\bbeta) = \by - \bX\bbeta$，$f(\mathbf{g}) = \mathbf{g}\T\mathbf{g}$）也得到同样的结果。令梯度为零，即得 {doc}`08-ols-and-ml-estimation` 中的正规方程。
:::

## 最优性条件与凸性

若对所有 $\mathbf{a},\mathbf{b}$ 及 $t\in[0,1]$ 有 $f\bigl(t\mathbf{a} + (1-t)\mathbf{b}\bigr) \le t f(\mathbf{a}) + (1-t)f(\mathbf{b})$，则称 $f$ 为**凸函数**；若当 $\mathbf{a}\ne\mathbf{b}$、$0<t<1$ 时不等式严格成立，则称其**严格凸**。若 $-f$ 是凸函数，则称 $f$ 为凹函数。

:::{prf:theorem} 最优性条件
:label: thm-optimality

设 $f$ 二阶连续可微。

1. 若 $f$ 在 $\bbeta^{*}$ 处取局部极小，则 $\nabla f(\bbeta^{*}) = \bzero$。
2. $f$ 为凸函数当且仅当其 Hessian 矩阵处处半正定；若 Hessian 矩阵处处正定，则 $f$ 严格凸。
3. 对凸函数 $f$，满足 $\nabla f(\bbeta^{*}) = \bzero$ 的点都是全局极小点；对严格凸函数，这样的点是唯一的极小点。
:::

最小二乘目标函数的 Hessian 矩阵为 $2\bX\T\bX$，因此目标函数是凸的，当 $\bX$ 列满秩时严格凸；此时 OLS 估计量是唯一的全局极小点。同理，使用典则联系函数的 GLM 对数似然是严格凹的（{doc}`17-generalized-linear-models`）。

### 二阶泰勒展开

在点 $\bbeta_0$ 附近，

$$
f(\bbeta) \approx f(\bbeta_0) + \nabla f(\bbeta_0)\T(\bbeta - \bbeta_0)
+ \tfrac12(\bbeta - \bbeta_0)\T\mathbf{H}(\bbeta_0)(\bbeta - \bbeta_0),
$$

其中 $\mathbf{H}$ 为 Hessian 矩阵。对右边求极小，得到 Newton–Raphson 迭代步 $\bbeta_1 = \bbeta_0 - \mathbf{H}(\bbeta_0)^{-1}\nabla f(\bbeta_0)$，它被用来拟合 GLM。

## 次梯度与绝对值函数

lasso 的惩罚项 $\lVert\bbeta\rVert_1 = \sum_j\lvert\beta_j\rvert$ 是凸函数，但在某个 $\beta_j = 0$ 处不可微。对凸函数 $f$，若对所有 $\mathbf{b}$ 有 $f(\mathbf{b}) \ge f(\bbeta) + \mathbf{g}\T(\mathbf{b} - \bbeta)$，则称向量 $\mathbf{g}$ 为 $f$ 在 $\bbeta$ 处的**次梯度**（subgradient）。所有次梯度构成的集合称为**次微分** $\partial f(\bbeta)$。对绝对值函数，

$$
\partial\lvert\beta\rvert =
\begin{cases}
\{1\}, & \beta > 0, \\
[-1, 1], & \beta = 0, \\
\{-1\}, & \beta < 0 .
\end{cases}
$$

:::{prf:theorem} 凸函数的最优性条件
:label: thm-subgradient-optimality

凸函数 $f$ 在 $\bbeta^{*}$ 处取得最小值，当且仅当 $\bzero \in \partial f(\bbeta^{*})$。
:::

当 $f$ 可微时，次微分只包含梯度，条件退化为 $\nabla f(\bbeta^{*}) = \bzero$。{doc}`16-ridge-and-lasso` 用这一条件推导 lasso 的软阈值解。

## 本章回顾
### 小结

- $\partial(\mathbf{a}\T\bbeta)/\partial\bbeta = \mathbf{a}$；对对称的 $\mathbf{A}$，$\partial(\bbeta\T\mathbf{A}\bbeta)/\partial\bbeta = 2\mathbf{A}\bbeta$。
- $\partial\lVert\by-\bX\bbeta\rVert_2^2/\partial\bbeta = -2\bX\T(\by-\bX\bbeta)$，Hessian 矩阵为 $2\bX\T\bX$。
- Hessian 矩阵半正定意味着凸，正定意味着严格凸；对凸函数，驻点就是全局极小点。
- $\lvert\beta\rvert$ 在零处的次微分为 $[-1,1]$；$\bzero\in\partial f(\bbeta^{*})$ 刻画了凸函数的极小点。

### 常见错误

- 混用布局：在本书采用的分母布局下，$\partial(\mathbf{a}\T\bbeta)/\partial\bbeta = \mathbf{a}$ 是列向量，而不是 $\mathbf{a}\T$。
- 对非对称的 $\mathbf{A}$ 使用 $2\mathbf{A}\bbeta$；一般法则是 $(\mathbf{A}+\mathbf{A}\T)\bbeta$。
- 未检查凸性或 Hessian 矩阵，就把梯度为零的点当作极小点。

## 参考文献

- Boyd, S., & Vandenberghe, L. (2004). *Convex optimization*. Cambridge University Press. <https://doi.org/10.1017/CBO9780511804441>
- Petersen, K. B., & Pedersen, M. S. (2012). *The matrix cookbook* (Version November 15, 2012). Technical University of Denmark. <https://www2.imm.dtu.dk/pubdb/pubs/3274-full.html>

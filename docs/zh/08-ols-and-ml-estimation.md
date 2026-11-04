# 最小二乘估计与极大似然估计

## 概述

本章推导系数向量 $\bbeta$ 的两种估计量：不需要任何分布假设的普通最小二乘（ordinary least squares, OLS）估计量，以及正态误差下的极大似然（maximum likelihood, ML）估计量。二者对 $\bbeta$ 的估计相同，但对误差方差 $\sigma^2$ 的估计不同，由此引出后续所有推断中使用的无偏估计量。 本章以 {doc}`07-linear-regression-model` 和向量求导（{doc}`06-matrix-calculus`）为基础。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 推导标量形式和矩阵形式的正规方程；
- 说明 OLS 估计量唯一的条件；
- 推导正态误差下 $\bbeta$ 和 $\sigma^2$ 的 ML 估计量；
- 解释为什么用 $\mathit{SS}_E/(n-p)$ 而不是 $\mathit{SS}_E/n$ 估计 $\sigma^2$。
:::

## 普通最小二乘

### 目标函数

**最小二乘**估计量选择使响应与线性预测之间偏差平方和最小的系数：

$$
S(\beta_0,\beta_1,\ldots,\beta_k)
= \sum_{i=1}^{n} \bigl(y_i - \beta_0 - \beta_1 x_{i1} - \cdots - \beta_k x_{ik}\bigr)^2 .
$$ (eq-ls-scalar)

为与加权或广义最小二乘相区别，该估计量称为**普通最小二乘（OLS）**。

### 标量形式的正规方程

令 {eq}`eq-ls-scalar` 的偏导数为零，得到含 $p$ 个未知数的 $p$ 个方程：

$$
\begin{gathered}
\frac{\partial S}{\partial \beta_0}
= -2\sum_{i=1}^{n}\bigl(y_i-\beta_0-\beta_1x_{i1}-\cdots-\beta_kx_{ik}\bigr) = 0, \\
\frac{\partial S}{\partial \beta_j}
= -2\sum_{i=1}^{n}\bigl(y_i-\beta_0-\beta_1x_{i1}-\cdots-\beta_kx_{ik}\bigr)\,x_{ij} = 0,
\qquad j = 1,\ldots,k .
\end{gathered}
$$

其解记为 $\widehat{\beta}_0,\widehat{\beta}_1,\ldots,\widehat{\beta}_k$，即 OLS 估计值。

### 矩阵形式

用矩阵表示，目标函数是偏差向量欧氏范数的平方：

$$
\begin{aligned}
S(\bbeta) &= \lVert \by - \bX\bbeta \rVert_2^2 \\
&= (\by - \bX\bbeta)\T(\by - \bX\bbeta), \\
\widehat{\bbeta} &= \operatorname*{arg\,min}_{\bbeta \in \R^p} S(\bbeta),
\end{aligned}
$$

其中 $\lVert\mathbf{a}\rVert_2 = (\mathbf{a}\T\mathbf{a})^{1/2}$ 为欧氏（$L_2$）范数。展开得

$$
\begin{aligned}
S(\bbeta)
&= \by\T\by - \by\T\bX\bbeta - \bbeta\T\bX\T\by + \bbeta\T\bX\T\bX\bbeta \\
&= \by\T\by - 2\bbeta\T\bX\T\by + \bbeta\T\bX\T\bX\bbeta ,
\end{aligned}
$$

这里用到 $\by\T\bX\bbeta$ 是标量，等于其转置 $\bbeta\T\bX\T\by$。

利用 {doc}`06-matrix-calculus` 中的向量求导法则 $\partial(\mathbf{a}\T\bbeta)/\partial\bbeta = \mathbf{a}$ 以及对称矩阵 $\mathbf{A}$ 满足的 $\partial(\bbeta\T\mathbf{A}\bbeta)/\partial\bbeta = 2\mathbf{A}\bbeta$，有

$$
\begin{aligned}
\frac{\partial S}{\partial \bbeta} &= -2\bX\T\by + 2\bX\T\bX\bbeta \\
&= \bzero .
\end{aligned}
$$

由此得到**正规方程**（normal equations）

$$
\bX\T\bX\,\widehat{\bbeta} = \bX\T\by .
$$ (eq-normal-equations)

若 $\bX\T\bX$ 非奇异，则唯一解为

$$
\widehat{\bbeta} = (\bX\T\bX)^{-1}\bX\T\by .
$$ (eq-ols)

在同一条件下，Hessian 矩阵 $\partial^2 S/\partial\bbeta\,\partial\bbeta\T = 2\bX\T\bX$ 正定，因此 {eq}`eq-ols` 是唯一的极小点。

:::{prf:lemma} Gram 矩阵的可逆性
:label: lem-gram-invertible

$p\times p$ 矩阵 $\bX\T\bX$ 非奇异，当且仅当 $\rank(\bX) = p$。
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

对任意 $\mathbf{a}\in\R^p$，有 $\mathbf{a}\T\bX\T\bX\mathbf{a} = \lVert\bX\mathbf{a}\rVert_2^2 \ge 0$，等号成立当且仅当 $\bX\mathbf{a} = \bzero$。若 $\rank(\bX) = p$，则 $\bX$ 的列线性无关，$\bX\mathbf{a} = \bzero$ 仅当 $\mathbf{a} = \bzero$，故 $\bX\T\bX$ 正定、可逆。反之，若 $\rank(\bX) < p$，则存在 $\mathbf{a}\ne\bzero$ 使 $\bX\mathbf{a} = \bzero$，从而 $\bX\T\bX\mathbf{a} = \bzero$，$\bX\T\bX$ 奇异。
:::

这就是 {prf:ref}`asm-classical` 要求列满秩的原因。

### 一元线性回归

只有一个预测变量时，$y_i = \beta_0 + \beta_1x_i + \varepsilon_i$，正规方程有闭式解

$$
\begin{gathered}
\widehat{\beta}_1 = \frac{S_{xy}}{S_{xx}}, \\
\widehat{\beta}_0 = \bar{y} - \widehat{\beta}_1\bar{x},
\end{gathered}
$$

其中

$$
\begin{aligned}
S_{xx} &= \sum_{i=1}^{n}(x_i - \bar{x})^2, \\
S_{xy} &= \sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y}) \\
&= \sum_{i=1}^{n}(x_i - \bar{x})\,y_i .
\end{aligned}
$$

## 极大似然估计

在正态性假设下，$\by \sim \mathcal{N}(\bX\bbeta, \sigma^2\bI_n)$。$(\bbeta,\sigma^2)$ 的似然函数就是 $\by$ 的联合密度：

$$
\begin{aligned}
\mathcal{L}(\bbeta,\sigma^2 \mid \by,\bX)
&= \frac{1}{\sqrt{(2\pi)^n \det(\sigma^2\bI_n)}}
   \exp\!\Bigl[-\tfrac12(\by-\bX\bbeta)\T(\sigma^2\bI_n)^{-1}(\by-\bX\bbeta)\Bigr] \\
&= (2\pi\sigma^2)^{-n/2}
   \exp\!\Bigl[-\frac{1}{2\sigma^2}(\by-\bX\bbeta)\T(\by-\bX\bbeta)\Bigr].
\end{aligned}
$$

对数似然函数为

$$
\ell(\bbeta,\sigma^2)
= -\frac{n}{2}\ln(2\pi\sigma^2)
  - \frac{1}{2\sigma^2}(\by-\bX\bbeta)\T(\by-\bX\bbeta).
$$ (eq-normal-loglik)

分别对 $\bbeta$ 和 $\sigma^2$ 求导：

$$
\begin{gathered}
\frac{\partial\ell}{\partial\bbeta}
= \frac{1}{\sigma^2}\bX\T(\by-\bX\bbeta) = \bzero \qquad \Longrightarrow\quad \bX\T\bX\bbeta = \bX\T\by, \\
\frac{\partial\ell}{\partial\sigma^2}
= -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}(\by-\bX\bbeta)\T(\by-\bX\bbeta) = 0 \qquad \Longrightarrow\quad (\by-\bX\bbeta)\T(\by-\bX\bbeta) = n\sigma^2 .
\end{gathered}
$$

第一个条件仍是正规方程 {eq}`eq-normal-equations`，因此 $\bbeta$ 的 ML 估计量与 OLS 估计量相同。将其代入第二个条件，得到误差方差的 ML 估计量：

$$
\begin{aligned}
\widehat{\bbeta}_{\mathrm{ML}} &= (\bX\T\bX)^{-1}\bX\T\by, \\
\widehat{\sigma}^2_{\mathrm{ML}} &= \frac{1}{n}(\by-\bX\widehat{\bbeta})\T(\by-\bX\widehat{\bbeta}) \\
&= \frac{1}{n}\lVert\by-\widehat{\by}\rVert_2^2 \\
&= \frac{\mathit{SS}_E}{n},
\end{aligned}
$$

其中 $\widehat{\by} = \bX\widehat{\bbeta}$ 为拟合值向量。

## 误差方差的估计

**残差平方和**（residual sum of squares，也称误差平方和）为

$$
\begin{aligned}
\mathit{SS}_E &= \sum_{i=1}^{n} e_i^2 \\
&= \lVert\be\rVert_2^2 \\
&= (\by-\bX\widehat{\bbeta})\T(\by-\bX\widehat{\bbeta}), \\
\be &= \by - \widehat{\by}.
\end{aligned}
$$

ML 估计量 $\mathit{SS}_E/n$ 是有偏的。推断中使用的估计量除以残差自由度 $n-p$：

$$
\begin{aligned}
\widehat{\sigma}^2 &= \frac{\mathit{SS}_E}{n-p} \\
&= \mathit{MS}_E ,
\end{aligned}
$$ (eq-sigma2-unbiased)

它满足 $\E[\widehat{\sigma}^2] = \sigma^2$（见 {doc}`09-ols-properties-and-geometry` 中的 {prf:ref}`thm-sigma2-unbiased`）。直观上，拟合 $p$ 个系数使残差满足 $p$ 个线性约束（$\bX\T\be = \bzero$），因此只有 $n-p$ 个残差可以自由变化。本书中不带下标的 $\widehat{\sigma}^2$ 均指无偏估计量 {eq}`eq-sigma2-unbiased`。

## 本章回顾
### 小结

- OLS 最小化 $\lVert\by-\bX\bbeta\rVert_2^2$，其解满足正规方程 $\bX\T\bX\widehat{\bbeta} = \bX\T\by$。
- 当 $\rank(\bX) = p$ 时，解 $\widehat{\bbeta} = (\bX\T\bX)^{-1}\bX\T\by$ 唯一。
- 在正态误差下，$\bbeta$ 的 ML 估计量等于 OLS 估计量。
- ML 方差估计量 $\mathit{SS}_E/n$ 有偏；$\widehat{\sigma}^2 = \mathit{SS}_E/(n-p)$ 无偏。

### 常见错误

- 将 OLS 目标函数写成 $\lVert\by-\bX\bbeta\rVert_2$ 而不是其平方。二者极小点相同，但上文的求导针对的是范数的平方。
- 在正规方程中混用 $\by\T\bX$（行向量）与 $\bX\T\by$（列向量）。
- 在 $t$ 和 $F$ 统计量中使用 $\widehat{\sigma}^2_{\mathrm{ML}} = \mathit{SS}_E/n$；精确分布要求使用 $\mathit{SS}_E/(n-p)$。

## 参考文献

- Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley.
- He, Z. [何志坚]. (n.d.). *数理统计讲义* [Lecture notes on mathematical statistics]. <https://bookdown.org/hezhijian/book/>

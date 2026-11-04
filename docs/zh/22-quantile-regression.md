# 分位数回归

## 概述

前面各章建模的都是条件**均值**：线性回归中的 $\E[Y\mid\bx]$ 和 GLM 中的 $g^{-1}(\bx\T\bbeta)$。均值只概括了条件分布的中心。**分位数回归**（quantile regression）转而对条件**分位数**建模，例如中位数或第 10 百分位数，从而允许响应分布的不同部分以不同方式依赖于预测变量。它用非对称的弹球损失代替最小二乘的平方误差损失。 本章以把最小二乘看作损失最小化（{doc}`08-ols-and-ml-estimation`）为基础。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 定义随机变量的 $\tau$ 分位数；
- 写出弹球（分位数）损失并解释其非对称性；
- 写出线性分位数回归模型及其估计问题；
- 比较分位数回归与条件均值回归。
:::

## 分位数

设随机变量 $X$ 的分布函数为 $F$，其 **$\tau$ 分位数**（$0 < \tau < 1$）是满足下式的值 $x_\tau$：

$$
\begin{aligned}
\tau &= \Prob(X \le x_\tau) \\
&= F(x_\tau).
\end{aligned}
$$

对有跳跃或平坦部分的分布，一般定义为 $x_\tau = \inf\{x : F(x) \ge \tau\}$。中位数就是 $0.5$ 分位数。

## 弹球损失

最小二乘最小化均方误差，即 $L_2$ 损失。分位数回归则对残差 $u = y - \widehat{y}$ 使用**弹球损失**（pinball loss，也称分位数损失或检验函数）$\rho_\tau(u)$：

$$
\rho_\tau(u) =
\begin{cases}
\tau\,u, & u \ge 0, \\
(\tau - 1)\,u, & u < 0,
\end{cases}
$$ (eq-pinball)

或等价地

$$
\rho_\tau(u) = \tau\max(u,0) + (1-\tau)\max(-u,0).
$$

两个分支都非负：正残差（低估）的权重为 $\tau$，负残差（高估）的权重为 $1-\tau$。关于 $q$ 极小化 $\E[\rho_\tau(Y - q)]$，得到的就是 $Y$ 的 $\tau$ 分位数。

## 分位数回归模型

对给定的 $\tau\in(0,1)$，线性分位数回归把条件 $\tau$ 分位数建模为 $Q_\tau(Y\mid\bx) = \bx\T\bbeta_\tau$，并通过下式估计系数向量：

$$
\begin{aligned}
\widehat{\bbeta}_\tau
&= \operatorname*{arg\,min}_{\bbeta_\tau}\ \sum_{i=1}^n \rho_\tau\bigl(y_i - \bx_i\T\bbeta_\tau\bigr) \\
&= \operatorname*{arg\,min}_{\bbeta_\tau}\ \Biggl[\ \sum_{i:\ y_i \ge \bx_i\T\bbeta_\tau}\tau\bigl(y_i - \bx_i\T\bbeta_\tau\bigr)
 + \sum_{i:\ y_i < \bx_i\T\bbeta_\tau}(\tau-1)\bigl(y_i - \bx_i\T\bbeta_\tau\bigr)\Biggr].
\end{aligned}
$$ (eq-quantile-regression)

每个 $\tau$ 都有自己的系数向量 $\bbeta_\tau$。

- 当 $\tau = 0.5$ 时，$\rho_{0.5}(u) = |u|/2$，问题成为**中位数回归**，也称最小一乘回归（least absolute deviations）：它最小化平均绝对误差，即 $L_1$ 损失。
- 较高的分位数对低估的惩罚更重，较低的分位数对高估的惩罚更重。对 $0.25$ 分位数，高估 $y$ 的权重为 $0.75$，低估的权重为 $0.25$，因此拟合线偏低；对 $0.75$ 分位数则正好相反，拟合线偏高。

### 条件均值与条件分位数

最小二乘估计响应分布的**中心**如何依赖于 $\bx$。在多个 $\tau$ 上拟合分位数回归，则可以展示整个分布如何随 $\bx$ 平移、扩散或变得偏斜。当响应的离散程度随预测变量变化（异方差）、关注分布尾部（例如出行时间的第 90 百分位数），或 $y$ 中的离群值会扭曲均值回归时，分位数回归都很有用；中位数回归对这类离群值比最小二乘更稳健。

### 估计

目标函数 {eq}`eq-quantile-regression` 是分段线性的，在残差为零处不可微，因此不存在像 {eq}`eq-ols` 那样的闭式解。该问题可以写成线性规划。主要的估计算法有单纯形法、内点法和平滑算法。

## 实现

用 `scikit-learn` 和 `statsmodels` 拟合分位数回归的方法见 {doc}`28-quantile-regression-python`。

## 本章回顾
### 小结

- $\tau$ 分位数是以弹球损失 $\rho_\tau$ 求解 $\min_q \E[\rho_\tau(Y-q)]$ 的结果。
- 线性分位数回归建模 $Q_\tau(Y\mid\bx) = \bx\T\bbeta_\tau$，并通过最小化总弹球损失估计 $\bbeta_\tau$。
- $\tau = 0.5$ 时为中位数（$L_1$）回归。
- 拟合多个分位数可以描述整个条件分布如何依赖于预测变量。

### 常见错误

- 把 $\bbeta_\tau$ 解释为对均值的效应；它是对条件 $\tau$ 分位数的效应。
- 期望在不同 $\tau$ 上分别拟合的分位数线永不相交；在有限样本中它们偶尔会相交。

## 参考文献

- Lianxh [连享会]. (n.d.). *Stata+R：分位数回归一文读懂* [Quantile regression with Stata and R]. <https://www.lianxh.cn/news/f29d83cd303dc.html>

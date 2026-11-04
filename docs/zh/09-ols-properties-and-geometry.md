# 最小二乘的几何意义与统计性质

## 概述

本章从两个角度研究最小二乘估计量。几何角度：拟合值是 $\by$ 在 $\bX$ 列空间上的正交投影。统计角度：推导 $\widehat{\bbeta}$ 的均值、协方差和抽样分布，并证明 $\widehat{\sigma}^2 = \mathit{SS}_E/(n-p)$ 是无偏的。这些结论是 {doc}`10-anova-and-inference` 中检验和区间估计的基础。 本章以 {doc}`08-ols-and-ml-estimation`、矩阵的迹与投影矩阵（{doc}`05-linear-algebra`）、随机向量的均值与协方差（{doc}`01-probability-and-random-vectors`）以及正态向量的二次型（{doc}`02-sampling-distributions-and-limit-theorems`）为基础。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 定义帽子矩阵，并证明它对称且幂等；
- 证明残差与拟合值以及 $\bX$ 的每一列正交；
- 推导 $\E[\widehat{\bbeta}]$ 和 $\Cov(\widehat{\bbeta})$；
- 证明 $\E[\mathit{SS}_E] = (n-p)\sigma^2$。
:::

## 帽子矩阵

将 OLS 估计量 {eq}`eq-ols` 代入 $\widehat{\by} = \bX\widehat{\bbeta}$，得

$$
\begin{aligned}
\widehat{\by} &= \bX(\bX\T\bX)^{-1}\bX\T\by \\
&= \bH\by, \\
\bH &= \bX(\bX\T\bX)^{-1}\bX\T \in \R^{n\times n}.
\end{aligned}
$$ (eq-hat-matrix)

矩阵 $\bH$ 称为**帽子矩阵**（hat matrix），因为它给 $\by$"戴上了帽子"。它也称为**投影矩阵**（projection matrix），因为它把 $\by$ 正交投影到 $\bX$ 的列空间上。残差向量为

$$
\begin{aligned}
\be &= \by - \widehat{\by} \\
&= (\bI_n - \bH)\by .
\end{aligned}
$$ (eq-residual-projection)

:::{prf:lemma} 帽子矩阵的性质
:label: lem-hat-properties

矩阵 $\bH$ 和 $\bI_n - \bH$ 都是对称幂等矩阵：

$$
\begin{aligned}
\bH &= \bH\T \\
&= \bH^2, \\
\bI_n - \bH &= (\bI_n - \bH)\T \\
&= (\bI_n - \bH)^2 .
\end{aligned}
$$

此外，$\bH\bX = \bX$，$(\bI_n-\bH)\bX = \mathbf{0}$，且 $\tr(\bH) = p$。
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

由于 $(\bX\T\bX)^{-1}$ 对称，$\bH\T = \bX\bigl((\bX\T\bX)^{-1}\bigr)\T\bX\T = \bH$。幂等性来自

$$
\begin{aligned}
\bH^2 &= \bX(\bX\T\bX)^{-1}\underbrace{\bX\T\bX(\bX\T\bX)^{-1}}_{\bI_p}\bX\T \\
&= \bH .
\end{aligned}
$$

于是 $(\bI_n-\bH)^2 = \bI_n - 2\bH + \bH^2 = \bI_n - \bH$，$\bI_n-\bH$ 的对称性是显然的。其次，$\bH\bX = \bX(\bX\T\bX)^{-1}\bX\T\bX = \bX$，故 $(\bI_n-\bH)\bX = \mathbf{0}$。最后，由迹的循环性质（{prf:ref}`lem-trace-cyclic`），

$$
\begin{aligned}
\tr(\bH) &= \tr\bigl(\bX(\bX\T\bX)^{-1}\bX\T\bigr) \\
&= \tr\bigl(\bX\T\bX(\bX\T\bX)^{-1}\bigr) \\
&= \tr(\bI_p) \\
&= p .
\end{aligned}
$$
:::

## 残差的正交性

:::{prf:proposition} 正交性
:label: prop-orthogonality

残差向量与 $\bX$ 的每一列以及拟合值向量都正交：

$$
\begin{gathered}
\bX\T\be = \bzero, \\
\widehat{\by}\T\be = 0 .
\end{gathered}
$$

若模型含截距，则残差之和也为零：$\sum_{i=1}^n e_i = 0$。
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

由 {eq}`eq-residual-projection` 和 {prf:ref}`lem-hat-properties`，$\bX\T\be = \bX\T(\bI_n-\bH)\by = \bigl((\bI_n-\bH)\bX\bigr)\T\by = \bzero$。对于拟合值，

$$
\begin{aligned}
\widehat{\by}\T\be &= (\bH\by)\T(\bI_n-\bH)\by \\
&= \by\T\bH(\bI_n-\bH)\by \\
&= \by\T(\bH-\bH^2)\by \\
&= 0 .
\end{aligned}
$$

当 $\bX$ 的第一列为全 1 向量 $\mathbf{1}_n$ 时，$\bX\T\be = \bzero$ 的第一个分量就是 $\mathbf{1}_n\T\be = \sum_i e_i = 0$。
:::

从几何上看，$\by$ 被分解为两个正交部分：位于 $\bX$ 列空间中的拟合向量 $\widehat{\by}$，以及位于其正交补空间中的残差向量 $\be$。正是这种正交性使 {doc}`10-anova-and-inference` 中的平方和分解成立。

## OLS 估计量的均值与协方差

本节的结论用到随机向量线性变换的法则（{prf:ref}`lem-linear-transform`）。

:::{prf:theorem} $\widehat{\bbeta}$ 的无偏性与协方差
:label: thm-beta-unbiased

在 {prf:ref}`asm-classical` 的假设 1–5 下，

$$
\begin{gathered}
\E[\widehat{\bbeta}] = \bbeta, \\
\Cov(\widehat{\bbeta}) = \sigma^2(\bX\T\bX)^{-1}.
\end{gathered}
$$
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

记 $\widehat{\bbeta} = \mathbf{A}\by$，其中常数矩阵 $\mathbf{A} = (\bX\T\bX)^{-1}\bX\T$，并利用 $\E[\by] = \bX\bbeta$、$\Cov(\by) = \sigma^2\bI_n$。由 {prf:ref}`lem-linear-transform`，

$$
\begin{aligned}
\E[\widehat{\bbeta}] &= (\bX\T\bX)^{-1}\bX\T\E[\by] \\
&= (\bX\T\bX)^{-1}\bX\T\bX\bbeta \\
&= \bbeta ,
\end{aligned}
$$

又因为 $\bX\T\bX$ 对称，

$$
\begin{aligned}
\Cov(\widehat{\bbeta}) &= (\bX\T\bX)^{-1}\bX\T\,(\sigma^2\bI_n)\,\bX(\bX\T\bX)^{-1} \\
&= \sigma^2(\bX\T\bX)^{-1} .
\end{aligned}
$$
:::

### 协方差矩阵的含义

记 $\mathbf{C} = (\bX\T\bX)^{-1}$，其元素为 $c_{jl}$，下标 $j,l = 0,1,\ldots,k$ 与系数对应。$\widehat{\bbeta}$ 的协方差矩阵包含各个估计量的全部方差和协方差：

$$
\begin{aligned}
\Cov(\widehat{\bbeta})
&= \E\Bigl[\bigl(\widehat{\bbeta}-\E[\widehat{\bbeta}]\bigr)\bigl(\widehat{\bbeta}-\E[\widehat{\bbeta}]\bigr)\T\Bigr] \\
&= \begin{bmatrix}
\Var(\widehat{\beta}_0) & \cdots & \Cov(\widehat{\beta}_0,\widehat{\beta}_k) \\
\vdots & \ddots & \vdots \\
\Cov(\widehat{\beta}_k,\widehat{\beta}_0) & \cdots & \Var(\widehat{\beta}_k)
\end{bmatrix} \\
&= \sigma^2
\begin{bmatrix}
c_{00} & \cdots & c_{0k} \\
\vdots & \ddots & \vdots \\
c_{k0} & \cdots & c_{kk}
\end{bmatrix}.
\end{aligned}
$$

特别地，$\Var(\widehat{\beta}_j) = \sigma^2 c_{jj}$，$\Cov(\widehat{\beta}_j,\widehat{\beta}_l) = \sigma^2 c_{jl}$。由于 $\Cov(\widehat{\beta}_j,\widehat{\beta}_l) = \Cov(\widehat{\beta}_l,\widehat{\beta}_j)$，该矩阵是对称的。

## 误差方差的无偏估计

:::{prf:theorem} $\widehat{\sigma}^2$ 的无偏性
:label: thm-sigma2-unbiased

在 {prf:ref}`asm-classical` 的假设 1–5 下，$\E[\mathit{SS}_E] = (n-p)\sigma^2$。因此 $\widehat{\sigma}^2 = \mathit{SS}_E/(n-p)$ 是 $\sigma^2$ 的无偏估计量，而 ML 估计量 $\mathit{SS}_E/n$ 偏小。
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

由 {eq}`eq-residual-projection` 和 {prf:ref}`lem-hat-properties`，

$$
\begin{aligned}
\mathit{SS}_E &= \be\T\be \\
&= \by\T(\bI_n-\bH)\T(\bI_n-\bH)\by \\
&= \by\T(\bI_n-\bH)\by .
\end{aligned}
$$

标量等于其自身的迹，于是由 {prf:ref}`lem-trace-cyclic` 以及期望与迹的线性性，

$$
\begin{aligned}
\E[\mathit{SS}_E]
&= \E\bigl[\tr\bigl(\by\T(\bI_n-\bH)\by\bigr)\bigr] \\
 &= \tr\bigl((\bI_n-\bH)\,\E[\by\by\T]\bigr) \\
&= \tr\bigl((\bI_n-\bH)\bigl(\Cov(\by) + \E[\by]\E[\by]\T\bigr)\bigr) \\
&= \tr\bigl((\bI_n-\bH)\,\sigma^2\bI_n\bigr)
 + \tr\bigl((\bI_n-\bH)\bX\bbeta\bbeta\T\bX\T\bigr) \\
&= \sigma^2\bigl(n - \tr(\bH)\bigr) + 0 \\
 &= \sigma^2(n-p),
\end{aligned}
$$

其中第二个迹为零是因为 $(\bI_n-\bH)\bX = \mathbf{0}$，而由 {prf:ref}`lem-hat-properties` 知 $\tr(\bH) = p$。
:::

## 正态误差下的抽样分布

加上正态性假设后，可以得到推断所用的精确分布。

:::{prf:theorem} OLS 估计量的分布
:label: thm-ols-distribution

在 {prf:ref}`asm-classical`（包括正态性）下：

1. $\widehat{\bbeta} \sim \mathcal{N}\bigl(\bbeta,\ \sigma^2(\bX\T\bX)^{-1}\bigr)$，特别地 $\widehat{\beta}_j \sim \mathcal{N}(\beta_j,\ \sigma^2 c_{jj})$；
2. $\dfrac{(n-p)\,\widehat{\sigma}^2}{\sigma^2} = \dfrac{\mathit{SS}_E}{\sigma^2} \sim \chi^2(n-p)$；
3. 残差向量 $\be$ 与 $\widehat{\by}$ 及 $\widehat{\bbeta}$ 相互独立；因此
4. $\mathit{SS}_E$（等价地，$\widehat{\sigma}^2$）与 $\widehat{\bbeta}$ 独立。
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

1. $\widehat{\bbeta} = (\bX\T\bX)^{-1}\bX\T\by$ 是正态向量 $\by$ 的线性变换，由 {prf:ref}`thm-mvn-properties` 知它服从正态分布，其均值和协方差由 {prf:ref}`thm-beta-unbiased` 给出。
2. 由于 $(\bI_n-\bH)\bX = \mathbf{0}$，残差向量 $\be = (\bI_n-\bH)\by = (\bI_n-\bH)\bvarepsilon$，因此

   $$
   \begin{gathered}
   \frac{\mathit{SS}_E}{\sigma^2} = \frac{\bvarepsilon\T(\bI_n-\bH)\bvarepsilon}{\sigma^2}, \\
   \bvarepsilon \sim \mathcal{N}_n(\bzero, \sigma^2\bI_n).
   \end{gathered}
   $$

   矩阵 $\bI_n-\bH$ 对称幂等，秩为 $\tr(\bI_n-\bH) = n-p$（{prf:ref}`prop-projection`），由 {prf:ref}`thm-quadratic-form-chi2` 即得 $\chi^2(n-p)$。
3. $\be = (\bI_n-\bH)\by$ 与 $\widehat{\bbeta} = (\bX\T\bX)^{-1}\bX\T\by$ 是 $\by$ 的联合正态线性函数，其互协方差为

   $$
   \begin{aligned}
   \Cov\bigl(\be,\ \widehat{\bbeta}\bigr)
   &= (\bI_n-\bH)\,(\sigma^2\bI_n)\,\bX(\bX\T\bX)^{-1} \\
   &= \sigma^2(\bI_n-\bH)\bX(\bX\T\bX)^{-1} \\
   &= \mathbf{0},
   \end{aligned}
   $$

   故由 {prf:ref}`thm-mvn-properties` 知二者独立；而 $\widehat{\by} = \bX\widehat{\bbeta}$ 是 $\widehat{\bbeta}$ 的函数。
4. $\mathit{SS}_E = \be\T\be$ 是 $\be$ 的函数，因此与 $\widehat{\bbeta}$ 独立。
:::

## 本章回顾
### 小结

- $\widehat{\by} = \bH\by$，帽子矩阵 $\bH$ 对称、幂等，迹为 $p$。
- 残差与 $\bX$ 的列以及 $\widehat{\by}$ 正交。
- $\widehat{\bbeta}$ 无偏，协方差为 $\sigma^2(\bX\T\bX)^{-1}$；$\Var(\widehat{\beta}_j) = \sigma^2 c_{jj}$。
- $\E[\mathit{SS}_E] = (n-p)\sigma^2$，因此 $\mathit{SS}_E/(n-p)$ 无偏。
- 在正态性下，$\widehat{\bbeta}$ 服从正态分布，$\mathit{SS}_E/\sigma^2 \sim \chi^2(n-p)$，且二者独立。

### 常见错误

- 认为 $\widehat{\by}\T\be = 0$ 需要截距。并不需要；只有 $\sum_i e_i = 0$ 需要全 1 列。
- 写成 $\E[\by] = \bX\by$；正确的是 $\E[\by] = \bX\bbeta$。
- 用 $\widehat{\sigma}^2$ 代替 $\sigma^2$ 后仍声称 $\widehat{\beta}_j$ 服从精确的正态分布。方差为估计值时，标准化后的估计量服从 $t$ 分布（{doc}`10-anova-and-inference`）。

## 参考文献

- Seber, G. A. F., & Lee, A. J. (2003). *Linear regression analysis* (2nd ed.). Wiley. <https://doi.org/10.1002/9780471722199>
- He, Z. [何志坚]. (n.d.). *数理统计讲义* [Lecture notes on mathematical statistics]. <https://bookdown.org/hezhijian/book/>
- Shedden, K. (2021). *Statistics 600: Regression analysis* [Lecture notes]. University of Michigan. <https://dept.stat.lsa.umich.edu/~kshedden/stats600/>

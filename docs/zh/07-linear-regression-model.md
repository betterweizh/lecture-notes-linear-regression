# 线性回归：模型与假设

## 概述

线性回归（linear regression）将连续型响应变量的条件均值表示为一组预测变量的线性函数。本章给出模型的标量形式和矩阵形式，确定第二部分通用的符号，并列出后续各章结论成立所需的假设。 本章假设读者已掌握期望、方差与正态分布（{doc}`01-probability-and-random-vectors`），以及转置、逆和秩等矩阵代数（{doc}`05-linear-algebra`）。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 写出多元线性回归模型的标量形式和矩阵形式；
- 说明 $\by$、$\bX$、$\bbeta$ 和 $\bvarepsilon$ 的维数；
- 区分参数、估计量、误差和残差；
- 列出经典正态线性模型的假设，并说明哪些结论依赖于哪些假设。
:::

## 符号说明

| 符号 | 类型与维数 | 含义 |
|---|---|---|
| $n$ | 整数 | 观测数 |
| $k$ | 整数 | 预测变量个数 |
| $p$ | 整数 | 系数个数，$p = k + 1$（含截距） |
| $y_i$ | 标量 | 第 $i$ 个观测的响应值，$i = 1,\ldots,n$ |
| $x_{ij}$ | 标量 | 第 $i$ 个观测第 $j$ 个预测变量的取值，$j = 1,\ldots,k$ |
| $\beta_j$ | 标量 | 回归系数，$j = 0,1,\ldots,k$ |
| $\varepsilon_i$ | 随机标量 | 第 $i$ 个观测不可观测的误差 |
| $\by$ | $\R^n$ 中的向量 | 响应向量 |
| $\bX$ | $\R^{n\times p}$ 中的矩阵 | 设计矩阵 |
| $\bbeta$ | $\R^p$ 中的向量 | 系数向量 |
| $\bvarepsilon$ | $\R^n$ 中的随机向量 | 误差向量 |
| $\widehat{\bbeta}$ | $\R^p$ 中的随机向量 | $\bbeta$ 的估计量 |
| $\widehat{y}_i$, $\widehat{\by}$ | 标量，$\R^n$ 中的向量 | 拟合值 |
| $e_i$, $\be$ | 标量，$\R^n$ 中的向量 | 残差，$\be = \by - \widehat{\by}$ |

## 模型定义

### 标量形式

对观测 $i = 1,\ldots,n$，多元线性回归模型为

$$
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_k x_{ik} + \varepsilon_i .
$$ (eq-mlr-scalar)

系数 $\beta_0,\beta_1,\ldots,\beta_k$ 是固定但未知的常数，称为**参数**或**回归系数**（regression coefficients）；$\beta_0$ 称为**截距**（intercept）。估计出系数之后，第 $i$ 个观测的**拟合值**（fitted value）为

$$
\widehat{y}_i = \widehat{\beta}_0 + \widehat{\beta}_1 x_{i1} + \cdots + \widehat{\beta}_k x_{ik} .
$$

模型含 $k$ 个预测变量和 $p = k + 1$ 个系数。若模型不含截距，则 $p = k$；这种情形在需要时单独讨论（例如 {doc}`10-anova-and-inference`）。

### 术语

同一个量在不同领域中有多种名称。

变量 $y$
: **响应变量**（response），也称因变量（dependent variable）或内生变量（endogenous variable）。

变量 $x_1,\ldots,x_k$
: **预测变量**（predictor），也称自变量（independent variable）、解释变量（explanatory variable）、回归变量（regressor）、协变量（covariate）或外生变量（exogenous variable）。

本书统一使用"响应变量"和"预测变量"。

### 哪些量是随机的？

预测变量 $x_{ij}$ 可能来自某个分布的抽样，但分析时以其观测值为条件，将其视为固定常数。误差 $\varepsilon_i$ 是随机变量。因此：

- 每个响应 $y_i$ 依赖于 $\varepsilon_i$，因而是随机变量；
- 估计量 $\widehat{\beta}_j$ 是响应的函数，因而也是随机变量；
- 拟合值 $\widehat{y}_i$ 是随机变量，用于估计条件均值 $\E[y_i \mid \bx_i]$。

**参数**（如 $\bbeta$）是固定的未知常数；**估计量**（如 $\widehat{\bbeta}$）是随机向量；**估计值**是由一组数据计算出的估计量的数值。

### 矩阵形式

将 $n$ 个方程 {eq}`eq-mlr-scalar` 堆叠起来，得到

$$
\by = \bX\bbeta + \bvarepsilon ,
$$ (eq-mlr-matrix)

其中

$$
\by = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}_{n\times 1},
\quad
\bX = \begin{bmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1k} \\
1 & x_{21} & x_{22} & \cdots & x_{2k} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \cdots & x_{nk}
\end{bmatrix}_{n\times p},
\quad
\bbeta = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_k \end{bmatrix}_{p\times 1},
\quad
\bvarepsilon = \begin{bmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_n \end{bmatrix}_{n\times 1}.
$$

矩阵 $\bX$ 称为**设计矩阵**（design matrix），其第一列全为 1，对应截距。$\bX$ 的第 $i$ 行为 $\bx_i\T = (1, x_{i1}, \ldots, x_{ik})$，于是 $y_i = \bx_i\T\bbeta + \varepsilon_i$。

## 模型假设

第二部分的结论依赖以下假设。不同结论所需的假设不同，各章会说明所用的假设。

:::{prf:assumption} 经典正态线性模型
:label: asm-classical

1. **线性。** 模型 {eq}`eq-mlr-matrix` 成立：$\E[\by] = \bX\bbeta$。
2. **列满秩。** $p < n$ 且 $\rank(\bX) = p$，即 $\bX$ 的任何一列都不是其他列的线性组合。
3. **零均值。** 对所有 $i$，$\E[\varepsilon_i] = 0$。
4. **同方差。** 对所有 $i$，$\Var(\varepsilon_i) = \sigma^2$，其中 $\sigma^2 > 0$ 未知。
5. **误差不相关。** 当 $i \neq j$ 时，$\Cov(\varepsilon_i, \varepsilon_j) = 0$。
6. **正态性。** $\varepsilon_i \sim \mathcal{N}(0,\sigma^2)$。
:::

用矩阵形式表示，假设 3–5 即

$$
\begin{gathered}
\E[\bvarepsilon] = \bzero, \\
\Var(\bvarepsilon) = \sigma^2 \bI_n ,
\end{gathered}
$$

再加上假设 6，得到 $\bvarepsilon \sim \mathcal{N}(\bzero, \sigma^2\bI_n)$。对于正态分布的误差，不相关等价于独立，所以假设 3–6 合起来表示误差独立同分布（i.i.d.）于 $\mathcal{N}(0,\sigma^2)$。

各项假设的作用不同：

- 假设 2 保证最小二乘估计量存在且唯一（{doc}`08-ols-and-ml-estimation`）；
- 假设 1–5 足以保证 $\widehat{\bbeta}$ 无偏，且协方差矩阵为 $\sigma^2(\bX\T\bX)^{-1}$（{doc}`09-ols-properties-and-geometry`）；
- 检验和区间估计中使用的精确 $t$ 分布和 $F$ 分布需要假设 6（{doc}`10-anova-and-inference`）。

## 本章回顾
### 小结

- 线性模型为 $\by = \bX\bbeta + \bvarepsilon$，其中 $\bX \in \R^{n\times p}$，$\bbeta \in \R^p$。
- 预测变量视为固定；响应、估计量、拟合值和残差都是随机的。
- 经典假设包括：线性、列满秩，以及误差零均值、同方差、不相关、服从正态分布。

### 常见错误

- 混淆误差与残差。误差 $\varepsilon_i$ 不可观测；残差 $e_i = y_i - \widehat{y}_i$ 由数据计算得到。
- $p$ 的计数不一致。本书中 $p$ 包括截距在内的全部待估系数。
- 认为预测变量本身必须服从正态分布。正态性假设只针对误差。

## 参考文献

- Heumann, C., Schomaker, M., & Shalabh. (2016). *Introduction to statistics and data analysis: With exercises, solutions and applications in R*. Springer. <https://doi.org/10.1007/978-3-319-46162-5>
- Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley.
- He, Z. [何志坚]. (n.d.). *数理统计讲义* [Lecture notes on mathematical statistics]. <https://bookdown.org/hezhijian/book/>

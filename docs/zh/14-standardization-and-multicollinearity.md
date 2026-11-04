# 标准化与多重共线性

## 概述

回归系数以对应预测变量的单位度量，因此不能直接比较其大小。对变量进行标准化后，所有系数处于同一尺度。标准化形式也最便于描述**多重共线性**（multicollinearity），即预测变量之间较强的线性关系，它会使系数估计量的方差膨胀。 本章以 $\widehat{\bbeta}$ 的协方差（{doc}`09-ols-properties-and-geometry`）以及 $t$ 检验和 $F$ 检验（{doc}`10-anova-and-inference`）为基础。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 拟合标准化回归，并将其系数换算回原始尺度；
- 说明多重共线性如何影响估计值、标准误和检验；
- 计算并解释方差膨胀因子（VIF）。
:::

## 标准化线性回归

### 变量的标准化

将每个变量减去样本均值，再除以样本标准差：

$$
\begin{gathered}
y_i^{*} = \frac{y_i - \bar{y}}{s_y}, \\
x_{ij}^{*} = \frac{x_{ij} - \bar{x}_j}{s_{x_j}}, \qquad i = 1,\ldots,n,\ \ j = 1,\ldots,k,
\end{gathered}
$$

其中 $s_y$ 和 $s_{x_j}$ 分别为响应变量 $y$ 和预测变量 $x_j$ 的样本标准差。标准化后的每个变量均值为零、方差为一。

### 标准化模型

将标准化响应对标准化预测变量回归，得到

$$
y_i^{*} = \beta_1^{*}x_{i1}^{*} + \beta_2^{*}x_{i2}^{*} + \cdots + \beta_k^{*}x_{ik}^{*} + \varepsilon_i^{*},
\qquad i = 1,\ldots,n .
$$

由于所有标准化变量的均值都为零，拟合平面经过原点，故模型不含截距。系数 $\beta_j^{*}$ 表示在其他预测变量保持不变时，$x_j$ 增加一个标准差，$y$ 的期望变化量（以 $y$ 的标准差为单位）。这些系数没有量纲，因此可以在不同预测变量之间比较。

### 尺度之间的换算

两种尺度下的 OLS 估计值满足

$$
\begin{gathered}
\widehat{\beta}_j^{*} = \widehat{\beta}_j\,\frac{s_{x_j}}{s_y}, \\
\widehat{\beta}_j = \widehat{\beta}_j^{*}\,\frac{s_y}{s_{x_j}}, \qquad j = 1,\ldots,k,
\end{gathered}
$$

原始模型的截距由 $\widehat{\beta}_0 = \bar{y} - \sum_{j=1}^k \widehat{\beta}_j\bar{x}_j$ 恢复。标准化只是线性的单位变换，因此换算回原尺度后的拟合值以及 $R^2$ 都不变。

### 相关矩阵形式

设 $\mathbf{Z}$ 为 $n\times k$ 矩阵，其元素为 $z_{ij} = (x_{ij}-\bar{x}_j)/\sqrt{S_{jj}}$，其中 $S_{jj} = \sum_{i=1}^n (x_{ij}-\bar{x}_j)^2$。则 $\mathbf{R}_{xx} = \mathbf{Z}\T\mathbf{Z}$ 是预测变量的 $k\times k$ **相关矩阵**。这种"单位长度"标准化与上面的标准化只差一个常数因子 $\sqrt{n-1}$；若响应变量也做同样的变换，得到的标准化系数相同。这种形式便于研究多重共线性。

## 多重共线性

### 问题所在

普通最小二乘并不要求预测变量互不相关，只要求 $\bX$ 列满秩（{prf:ref}`lem-gram-invertible`）。但当某些预测变量高度相关时，就称存在**多重共线性**，此时会出现两个问题。

1. **方差膨胀。** 多重共线性降低系数估计的精度：标准误 $\mathrm{SE}(\widehat{\beta}_j)$ 增大，$t$ 统计量变小。于是即使 $\beta_j \ne 0$，也可能无法拒绝 $H_0:\beta_j = 0$；检验的**功效**（power），即检测出非零系数的概率，会降低。
2. **数值不稳定。** 当 $\bX$ 的列近似线性相关时，$\bX\T\bX$ 近似奇异（病态），其逆在数值上不稳定。在完全共线时，$\bX\T\bX$ 不可逆，$\widehat{\bbeta}$ 不唯一。

两两相关系数可以发现两个预测变量之间的线性关系，但无法揭示三个或更多预测变量之间的近似线性相关。方差膨胀因子解决了这个问题。

### 方差膨胀因子

记 $R_j^2$ 为以 $x_j$ 为响应、对其余 $k-1$ 个预测变量（含截距）回归所得的决定系数：

$$
x_{ij} = \gamma_0 + \sum_{l \ne j}\gamma_l x_{il} + u_i .
$$

原模型中斜率估计量的方差可写为

$$
\Var(\widehat{\beta}_j) = \frac{\sigma^2}{S_{jj}}\cdot\frac{1}{1-R_j^2},
\qquad j = 1,\ldots,k .
$$

在相关矩阵形式下，$\mathbf{R}_{xx}^{-1}$ 的第 $j$ 个对角元恰好等于 $1/(1-R_j^2)$。因子

$$
\mathrm{VIF}_j = \frac{1}{1-R_j^2}, \qquad j = 1,\ldots,k,
$$ (eq-vif)

称为**方差膨胀因子**（variance inflation factor, VIF）：它是 $\Var(\widehat{\beta}_j)$ 相对于理想情形（$x_j$ 与其他预测变量不相关，$R_j^2 = 0$）的膨胀倍数。$x_j$ 对其他变量的线性依赖越强，$R_j^2$ 越接近 1，VIF 越大。

常用阈值：

- VIF 超过 $5$ 或 $10$（即 $R_j^2 > 0.8$ 或 $0.9$）表明共线性已成问题（James et al., 2013, p. 101）；
- Montgomery 和 Runger（2018, p. 350）建议 VIF 不应超过 $10$，按更严格的标准不应超过 $4$ 或 $5$。

当 VIF 较大时，可能的处理方法包括：删除或合并冗余的预测变量、收集更多数据，或使用岭回归等惩罚方法（{doc}`16-ridge-and-lasso`）。计算 VIF 的代码见 {doc}`24-linear-regression-python` 和 {doc}`25-linear-regression-r`。

### 说明

- 多重共线性主要影响单个系数。即使某些 $\widehat{\beta}_j$ 的 VIF 很大，拟合值 $\widehat{y}$ 以及数据范围内的预测仍可能很准确。
- 典型症状是：整体 $F$ 检验显著，但若干单个系数的 $t$ 检验不显著。

## 本章回顾
### 小结

- 标准化系数 $\beta_j^{*} = \beta_j s_{x_j}/s_y$ 没有量纲，可以在预测变量之间比较。
- 多重共线性使 $\Var(\widehat{\beta}_j)$ 膨胀 $\mathrm{VIF}_j = 1/(1-R_j^2)$ 倍。
- VIF 超过约 $5$–$10$ 表明存在有害的共线性。

### 常见错误

- 通过比较单位不同的预测变量的原始系数来判断其重要性。
- 将系数换算到标准化尺度时用错因子：应为 $s_{x_j}/s_y$，而不是 $s_{x_1}/s_y$。
- 把两两相关系数当作多重共线性的完整检查。

## 参考文献

- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An introduction to statistical learning: With applications in R*. Springer. <https://doi.org/10.1007/978-1-4614-7138-7>
- Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley.

# 线性代数回顾

## 概述

线性模型以矩阵形式表述，其主要结论都是关于秩、逆、迹、投影和二次型的命题。本章汇集书中用到的线性代数知识：记号、秩与可逆性、迹、特征值与谱分解、正定矩阵、正交投影、奇异值分解以及向量范数。 本章假设读者已掌握基本的矩阵运算：加法、乘法和转置。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 将矩阵的秩与其 Gram 矩阵的可逆性联系起来；
- 运用迹的循环性质；
- 陈述对称矩阵的谱分解，并刻画正定（半正定）矩阵；
- 识别正交投影矩阵，并证明其秩等于迹；
- 运用奇异值分解以及 $L_1$、$L_2$ 范数。
:::

## 记号

向量用粗体小写字母表示，且为列向量，如 $\bx\in\R^m$；矩阵用粗体大写字母表示，如 $\mathbf{A}\in\R^{m\times n}$，其元素为 $a_{ij}$。转置记为 $\mathbf{A}\T$，$m\times m$ 单位矩阵记为 $\bI_m$，$\mathbf{1}_m$ 为全 1 向量，$\mathbf{0}$ 为零向量或零矩阵。若方阵满足 $\mathbf{A} = \mathbf{A}\T$，则称其**对称**。对可乘的矩阵，$(\mathbf{A}\mathbf{B})\T = \mathbf{B}\T\mathbf{A}\T$；当两个逆都存在时，$(\mathbf{A}\mathbf{B})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$。

$\mathbf{a},\mathbf{b}\in\R^m$ 的**内积**为 $\mathbf{a}\T\mathbf{b} = \sum_i a_ib_i$；若 $\mathbf{a}\T\mathbf{b} = 0$，则称两向量**正交**。

## 向量范数

| 范数 | 定义 | 用途 |
|---|---|---|
| $L_2$（欧氏范数） | $\lVert\mathbf{a}\rVert_2 = \bigl(\sum_i a_i^2\bigr)^{1/2} = (\mathbf{a}\T\mathbf{a})^{1/2}$ | 最小二乘、岭回归 |
| $L_1$ | $\lVert\mathbf{a}\rVert_1 = \sum_i \lvert a_i\rvert$ | lasso 回归、最小一乘 |
| $L_\infty$ | $\lVert\mathbf{a}\rVert_\infty = \max_i \lvert a_i\rvert$ | |

$L_2$ 范数的平方是光滑的，而 $L_1$ 范数在某个分量为零处不可微；这一差别解释了为什么 lasso 能把系数精确地压缩为零，而岭回归不能（{doc}`16-ridge-and-lasso`）。

## 秩与可逆性

$\mathbf{A}\in\R^{m\times n}$ 的**列空间**为 $\mathcal{C}(\mathbf{A}) = \{\mathbf{A}\mathbf{v} : \mathbf{v}\in\R^n\}$。$\mathbf{A}$ 的**秩**是其列空间的维数，即线性无关列（等价地，行）的个数。若 $\rank(\mathbf{A}) = n$，则称 $\mathbf{A}$ **列满秩**。

常用结论：

- $\rank(\mathbf{A}) \le \min(m, n)$，$\rank(\mathbf{A}\mathbf{B}) \le \min\bigl(\rank(\mathbf{A}), \rank(\mathbf{B})\bigr)$；
- $\rank(\mathbf{A}\T\mathbf{A}) = \rank(\mathbf{A})$；
- 方阵 $\mathbf{A}\in\R^{n\times n}$ **可逆**（非奇异）当且仅当 $\rank(\mathbf{A}) = n$，当且仅当 $\det(\mathbf{A}) \ne 0$。

因此，设计矩阵 $\bX\in\R^{n\times p}$ 的 Gram 矩阵 $\bX\T\bX$ 可逆，恰好当 $\bX$ 列满秩时成立（{prf:ref}`lem-gram-invertible`）。

## 迹

方阵的**迹**是其对角元之和，$\tr(\mathbf{A}) = \sum_i a_{ii}$。迹是线性的，$\tr(\mathbf{A} + \mathbf{B}) = \tr(\mathbf{A}) + \tr(\mathbf{B})$，$\tr(c\mathbf{A}) = c\,\tr(\mathbf{A})$，并且 $\tr(\mathbf{A}\T) = \tr(\mathbf{A})$。

:::{prf:lemma} 迹的循环性质
:label: lem-trace-cyclic

若 $\mathbf{A}$ 为 $m\times n$ 矩阵，$\mathbf{B}$ 为 $n\times m$ 矩阵，则 $\tr(\mathbf{A}\mathbf{B}) = \tr(\mathbf{B}\mathbf{A})$。
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

$$
\begin{aligned}
\tr(\mathbf{A}\mathbf{B}) &= \sum_{i=1}^m\sum_{j=1}^n a_{ij}b_{ji} \\
&= \sum_{j=1}^n\sum_{i=1}^m b_{ji}a_{ij} \\
&= \tr(\mathbf{B}\mathbf{A}).
\end{aligned}
$$
:::

特别地，标量二次型满足 $\mathbf{a}\T\mathbf{M}\mathbf{a} = \tr(\mathbf{M}\mathbf{a}\mathbf{a}\T)$。

## 特征值与谱分解

若标量 $\lambda$ 与非零向量 $\mathbf{v}$ 满足 $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$，则称它们为方阵 $\mathbf{A}$ 的一对**特征值–特征向量**。迹等于特征值之和，行列式等于特征值之积。

:::{prf:theorem} 谱分解
:label: thm-spectral

对称矩阵 $\mathbf{A}\in\R^{m\times m}$ 有实特征值 $\lambda_1,\ldots,\lambda_m$ 和一组标准正交的特征向量 $\mathbf{q}_1,\ldots,\mathbf{q}_m$，使得

$$
\begin{aligned}
\mathbf{A} &= \mathbf{Q}\boldsymbol\Lambda\mathbf{Q}\T \\
&= \sum_{j=1}^m \lambda_j\mathbf{q}_j\mathbf{q}_j\T,
\end{aligned}
$$

其中 $\mathbf{Q} = [\mathbf{q}_1,\ldots,\mathbf{q}_m]$ 为正交矩阵（$\mathbf{Q}\T\mathbf{Q} = \mathbf{Q}\mathbf{Q}\T = \bI_m$），$\boldsymbol\Lambda = \diag(\lambda_1,\ldots,\lambda_m)$。
:::

## 正定矩阵与二次型

对对称矩阵 $\mathbf{A}$，函数 $\mathbf{a}\mapsto\mathbf{a}\T\mathbf{A}\mathbf{a}$ 称为**二次型**。称 $\mathbf{A}$ 为

- **正定**的，若对所有 $\mathbf{a}\ne\mathbf{0}$ 有 $\mathbf{a}\T\mathbf{A}\mathbf{a} > 0$，等价于所有特征值为正；
- **半正定**的，若对所有 $\mathbf{a}$ 有 $\mathbf{a}\T\mathbf{A}\mathbf{a} \ge 0$，等价于所有特征值非负。

例如：

- 任何协方差矩阵都是半正定的；
- $\bX\T\bX$ 是半正定的，因为 $\mathbf{a}\T\bX\T\bX\mathbf{a} = \lVert\bX\mathbf{a}\rVert_2^2 \ge 0$；当 $\bX$ 列满秩时它是正定的；
- 对任意 $\lambda > 0$，即使 $\bX$ 不是列满秩，$\bX\T\bX + \lambda\bI_p$ 也是正定的，因而可逆。这是岭回归的代数基础。

## 正交投影与幂等矩阵

若方阵 $\mathbf{P}$ 满足 $\mathbf{P}^2 = \mathbf{P}$，则称其**幂等**。对称幂等矩阵是**正交投影矩阵**：它把每个向量映射到子空间 $\mathcal{C}(\mathbf{P})$ 上，余下的部分 $(\bI - \mathbf{P})\mathbf{a}$ 与该子空间正交。

:::{prf:proposition} 投影矩阵的性质
:label: prop-projection

设 $\mathbf{P}$ 对称且幂等，则

1. 其特征值为 $0$ 或 $1$；
2. $\rank(\mathbf{P}) = \tr(\mathbf{P})$；
3. $\bI - \mathbf{P}$ 也是对称幂等矩阵，且 $\rank(\bI - \mathbf{P}) = m - \rank(\mathbf{P})$。
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

若 $\mathbf{P}\mathbf{v} = \lambda\mathbf{v}$，$\mathbf{v}\ne\mathbf{0}$，则 $\lambda\mathbf{v} = \mathbf{P}\mathbf{v} = \mathbf{P}^2\mathbf{v} = \lambda^2\mathbf{v}$，故 $\lambda\in\{0,1\}$。由 {prf:ref}`thm-spectral`，秩等于非零特征值的个数，即 1 的个数，也就是特征值之和，即迹。对第 3 条，$(\bI-\mathbf{P})^2 = \bI - 2\mathbf{P} + \mathbf{P}^2 = \bI - \mathbf{P}$，其迹为 $m - \tr(\mathbf{P})$。
:::

帽子矩阵 $\bH = \bX(\bX\T\bX)^{-1}\bX\T$ 投影到 $\bX$ 的列空间上，其秩为 $\tr(\bH) = p$（{prf:ref}`lem-hat-properties`）。中心化矩阵 $\bI_n - \frac1n\mathbf{1}_n\mathbf{1}_n\T$ 投影到分量之和为零的向量构成的子空间上，其秩为 $n-1$。

## 奇异值分解

:::{prf:theorem} 奇异值分解（SVD）
:label: thm-svd

任何 $n \ge p$ 的矩阵 $\bX\in\R^{n\times p}$ 都可以写成

$$
\bX = \mathbf{U}\mathbf{D}\mathbf{V}\T,
$$

其中 $\mathbf{U}\in\R^{n\times p}$ 的列标准正交，$\mathbf{V}\in\R^{p\times p}$ 为正交矩阵，$\mathbf{D} = \diag(d_1,\ldots,d_p)$，$d_1 \ge \cdots \ge d_p \ge 0$ 为 $\bX$ 的**奇异值**。
:::

由此 $\bX\T\bX = \mathbf{V}\mathbf{D}^2\mathbf{V}\T$，即 $\bX\T\bX$ 的特征值为 $d_j^2$。较小的奇异值意味着 $\bX\T\bX$ 近似奇异，这是多重共线性的代数表现（{doc}`14-standardization-and-multicollinearity`）；SVD 也解释了岭回归如何收缩拟合值（{doc}`16-ridge-and-lasso`）。

## 本章回顾
### 小结

- $\bX\T\bX$ 可逆当且仅当 $\bX$ 列满秩。
- $\tr(\mathbf{A}\mathbf{B}) = \tr(\mathbf{B}\mathbf{A})$。
- 对称矩阵有标准正交的特征基；正定意味着所有特征值为正。
- 对称幂等矩阵是正交投影，其特征值为 $0$ 和 $1$，秩等于迹。
- $\bX = \mathbf{U}\mathbf{D}\mathbf{V}\T$，$\bX\T\bX$ 的特征值是奇异值的平方。

### 常见错误

- 写成 $(\mathbf{A}\mathbf{B})\T = \mathbf{A}\T\mathbf{B}\T$；顺序应当颠倒。
- 使用 $\tr(\mathbf{A}\mathbf{B}\mathbf{C}) = \tr(\mathbf{B}\mathbf{A}\mathbf{C})$；只允许循环置换。
- 认为只要 $\bX$ 的行数多于列数，$\bX\T\bX$ 就可逆；还需要各列线性无关。

## 参考文献

- Harville, D. A. (1997). *Matrix algebra from a statistician's perspective*. Springer. <https://doi.org/10.1007/b98818>
- Seber, G. A. F., & Lee, A. J. (2003). *Linear regression analysis* (2nd ed.). Wiley. <https://doi.org/10.1002/9780471722199>

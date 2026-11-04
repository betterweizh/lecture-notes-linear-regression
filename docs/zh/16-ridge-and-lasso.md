# 岭回归与 lasso 回归

## 概述

OLS 估计量是无偏的，但当预测变量高度相关或数量很多时，其方差可能非常大（{doc}`14-standardization-and-multicollinearity`）。**惩罚（正则化）回归**（penalized regression）最小化残差平方和加上对系数大小的惩罚。惩罚项以少量偏差换取方差的大幅降低，从而可以减小估计和预测的均方误差。**岭回归**（ridge regression）使用 $L_2$ 范数平方惩罚，具有闭式解；**lasso** 使用 $L_1$ 惩罚，会把部分系数精确地压缩为零，因而同时进行变量选择（{doc}`15-variable-selection`）。 本章以 {doc}`09-ols-properties-and-geometry`、偏差–方差分解（{doc}`02-sampling-distributions-and-limit-theorems`）、SVD 与范数（{doc}`05-linear-algebra`）以及梯度与次梯度（{doc}`06-matrix-calculus`）为基础。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 写出惩罚最小二乘目标函数（平方误差加惩罚项）及其约束形式；
- 推导岭估计量及其偏差、协方差和均方误差；
- 用奇异值分解解释岭回归如何收缩拟合，并计算其有效自由度；
- 推导正交设计下的 lasso 解（软阈值）以及坐标下降的更新公式；
- 解释为什么 lasso 产生稀疏解而岭回归不会；
- 用交叉验证选择惩罚参数。
:::

## 符号说明

| 符号 | 含义 |
|---|---|
| $\bX \in \R^{n\times k}$ | 已中心化和标准化的预测变量矩阵（不含全 1 列） |
| $\by \in \R^n$ | 已中心化的响应 |
| $\bbeta \in \R^k$ | 斜率系数 |
| $\lambda \ge 0$ | 惩罚（调节）参数 |
| $\widehat{\bbeta}^{\mathrm{OLS}}$, $\widehat{\bbeta}^{R}_\lambda$, $\widehat{\bbeta}^{L}_\lambda$ | OLS、岭回归和 lasso 估计量 |
| $\bX = \mathbf{U}\mathbf{D}\mathbf{V}\T$ | 奇异值分解，$\mathbf{D} = \diag(d_1,\ldots,d_k)$ |
| $S(z,\gamma) = \sign(z)\,(\lvert z\rvert - \gamma)_+$ | 软阈值算子，$(u)_+ = \max(u,0)$ |

## 带惩罚项的平方误差

### 标准化与截距

惩罚项对所有系数一视同仁，因此预测变量必须处于同一尺度。本章中每个预测变量都已中心化并缩放为单位方差，响应也已中心化（{doc}`14-standardization-and-multicollinearity`）。截距**不**受惩罚：中心化之后其估计为 $\widehat{\beta}_0 = \bar{y}$，余下的问题只涉及斜率。原始尺度上的系数通过逆标准化恢复。

### 惩罚最小二乘

惩罚最小二乘估计量最小化平方误差与惩罚项之和：

$$
\widehat{\bbeta}_\lambda
= \operatorname*{arg\,min}_{\bbeta\in\R^k}
\Bigl\{\underbrace{\lVert\by - \bX\bbeta\rVert_2^2}_{\text{平方误差}}
 + \underbrace{\lambda\,P(\bbeta)}_{\text{惩罚项}}\Bigr\}.
$$ (eq-penalized-ls)

| 方法 | 惩罚 $P(\bbeta)$ | 作用 |
|---|---|---|
| 岭回归 | $\lVert\bbeta\rVert_2^2 = \sum_{j=1}^k\beta_j^2$ | 把所有系数向零收缩 |
| lasso | $\lVert\bbeta\rVert_1 = \sum_{j=1}^k\lvert\beta_j\rvert$ | 收缩系数，并把部分系数精确地压缩为零 |
| 弹性网 | $\rho\lVert\bbeta\rVert_1 + (1-\rho)\lVert\bbeta\rVert_2^2$，$0\le\rho\le1$ | 兼有二者 |

调节参数 $\lambda$ 控制权衡：$\lambda = 0$ 时为 OLS，$\lambda\to\infty$ 时 $\widehat{\bbeta}_\lambda \to \bzero$。

### 约束形式

问题 {eq}`eq-penalized-ls` 等价于约束问题

$$
\min_{\bbeta}\ \lVert\by - \bX\bbeta\rVert_2^2
\qquad\text{约束条件}\qquad
P(\bbeta) \le t,
$$

即对每个 $\lambda \ge 0$ 都存在一个预算 $t \ge 0$ 使两者的解相同，反之亦然；$\lambda$ 是约束的拉格朗日乘子。约束形式提供了下文解释稀疏性所用的几何图景。

## 岭回归

### 岭估计量

:::{prf:theorem} 岭估计量
:label: thm-ridge

当 $\lambda > 0$ 时，岭回归目标函数 $\lVert\by - \bX\bbeta\rVert_2^2 + \lambda\lVert\bbeta\rVert_2^2$ 有唯一极小点

$$
\widehat{\bbeta}^{R}_\lambda = (\bX\T\bX + \lambda\bI_k)^{-1}\bX\T\by .
$$ (eq-ridge)
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

由 {prf:ref}`ex-gradient-rss` 以及 {doc}`06-matrix-calculus` 中的法则，

$$
\begin{aligned}
\frac{\partial}{\partial\bbeta}\Bigl[\lVert\by - \bX\bbeta\rVert_2^2 + \lambda\bbeta\T\bbeta\Bigr]
&= -2\bX\T(\by - \bX\bbeta) + 2\lambda\bbeta \\
&= -2\bX\T\by + 2(\bX\T\bX + \lambda\bI_k)\bbeta .
\end{aligned}
$$

令其为零得 $(\bX\T\bX + \lambda\bI_k)\bbeta = \bX\T\by$。当 $\lambda > 0$ 时 Hessian 矩阵 $2(\bX\T\bX + \lambda\bI_k)$ 正定（{doc}`05-linear-algebra`），因此该矩阵可逆且目标函数严格凸；由 {prf:ref}`thm-optimality`，该解是唯一的极小点。
:::

与 OLS 不同，即使 $\bX\T\bX$ 奇异（例如完全共线或 $k > n$），岭估计量也存在。

### 偏差、协方差与均方误差

记 $\mathbf{W}_\lambda = (\bX\T\bX + \lambda\bI_k)^{-1}\bX\T\bX$，当 $\bX$ 列满秩时 $\widehat{\bbeta}^{R}_\lambda = \mathbf{W}_\lambda\widehat{\bbeta}^{\mathrm{OLS}}$。

:::{prf:proposition} 岭估计量的矩
:label: prop-ridge-moments

在 {prf:ref}`asm-classical` 的假设 1–5 下，

$$
\begin{gathered}
\E\bigl[\widehat{\bbeta}^{R}_\lambda\bigr] = (\bX\T\bX + \lambda\bI_k)^{-1}\bX\T\bX\,\bbeta, \\
\operatorname{Bias}\bigl(\widehat{\bbeta}^{R}_\lambda\bigr) = -\lambda\,(\bX\T\bX + \lambda\bI_k)^{-1}\bbeta, \\
\Cov\bigl(\widehat{\bbeta}^{R}_\lambda\bigr) = \sigma^2(\bX\T\bX + \lambda\bI_k)^{-1}\bX\T\bX\,(\bX\T\bX + \lambda\bI_k)^{-1}.
\end{gathered}
$$
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

令 $\mathbf{A}_\lambda = (\bX\T\bX + \lambda\bI_k)^{-1}\bX\T$，则 $\widehat{\bbeta}^{R}_\lambda = \mathbf{A}_\lambda\by$。由 {prf:ref}`lem-linear-transform`，$\E[\widehat{\bbeta}^{R}_\lambda] = \mathbf{A}_\lambda\bX\bbeta$，$\Cov(\widehat{\bbeta}^{R}_\lambda) = \sigma^2\mathbf{A}_\lambda\mathbf{A}_\lambda\T$，即第一和第三个结果。对偏差，

$$
\begin{aligned}
\E\bigl[\widehat{\bbeta}^{R}_\lambda\bigr] - \bbeta
&= (\bX\T\bX + \lambda\bI_k)^{-1}\bigl[\bX\T\bX - (\bX\T\bX + \lambda\bI_k)\bigr]\bbeta \\
&= -\lambda\,(\bX\T\bX + \lambda\bI_k)^{-1}\bbeta .
\end{aligned}
$$
:::

岭估计量向零有偏，但其协方差比 OLS 小：在 $\bX\T\bX$ 的特征基下，每个方差都乘以 $d_j^4/(d_j^2+\lambda)^2 < 1$。奇异值分解把这种权衡表达得很清楚。

:::{prf:theorem} 岭回归的均方误差
:label: thm-ridge-mse

设 $\bX = \mathbf{U}\mathbf{D}\mathbf{V}\T$，$d_1,\ldots,d_k > 0$，并令 $\boldsymbol\theta = \mathbf{V}\T\bbeta$，则

$$
\begin{aligned}
\E\bigl[\lVert\widehat{\bbeta}^{R}_\lambda - \bbeta\rVert_2^2\bigr]
&= \underbrace{\sigma^2\sum_{j=1}^k\frac{d_j^2}{(d_j^2+\lambda)^2}}_{\text{总方差}}
 + \underbrace{\lambda^2\sum_{j=1}^k\frac{\theta_j^2}{(d_j^2+\lambda)^2}}_{\text{偏差平方}} .
\end{aligned}
$$ (eq-ridge-mse)

当 $\lambda = 0$ 时它等于 $\sigma^2\sum_j d_j^{-2}$，即 OLS 的总方差。它在 $\lambda = 0$ 处关于 $\lambda$ 的导数为 $-2\sigma^2\sum_j d_j^{-4} < 0$，因此总存在某个 $\lambda > 0$，使岭回归的均方误差小于 OLS（Hoerl & Kennard, 1970）。
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

由于 $\bX\T\bX = \mathbf{V}\mathbf{D}^2\mathbf{V}\T$，$\bX\T\bX + \lambda\bI_k = \mathbf{V}(\mathbf{D}^2 + \lambda\bI_k)\mathbf{V}\T$，由 {prf:ref}`prop-ridge-moments` 得

$$
\begin{gathered}
\Cov\bigl(\widehat{\bbeta}^{R}_\lambda\bigr)
= \sigma^2\mathbf{V}\,\diag\!\left(\frac{d_j^2}{(d_j^2+\lambda)^2}\right)\mathbf{V}\T, \\
\operatorname{Bias}\bigl(\widehat{\bbeta}^{R}_\lambda\bigr)
= -\lambda\,\mathbf{V}\,\diag\!\left(\frac{1}{d_j^2+\lambda}\right)\boldsymbol\theta .
\end{gathered}
$$

由 {prf:ref}`prop-mse-decomposition`，均方误差等于协方差矩阵的迹加上偏差向量长度的平方；由于 $\mathbf{V}$ 是正交矩阵，二者都可以在旋转后的坐标中计算，即得 {eq}`eq-ridge-mse`。求导可知，第一个和式的导数为 $-2\sigma^2\sum_j d_j^2/(d_j^2+\lambda)^3$，在 $\lambda = 0$ 处等于 $-2\sigma^2\sum_j d_j^{-4}$；第二个和式的导数 $2\lambda\sum_j d_j^2\theta_j^2/(d_j^2+\lambda)^3$ 在 $\lambda = 0$ 处为零。
:::

方差项随 $\lambda$ 增大而减小，偏差平方项随之增大；最优的 $\lambda$ 在二者之间取得平衡。当某些 $d_j$ 很小，即存在多重共线性时，收益最大，因为 OLS 的方差中含有 $1/d_j^2$。

### 收缩与有效自由度

岭回归的拟合值为

$$
\begin{aligned}
\widehat{\by}^{R}_\lambda
&= \bX(\bX\T\bX + \lambda\bI_k)^{-1}\bX\T\by \\
&= \sum_{j=1}^k \mathbf{u}_j\,\frac{d_j^2}{d_j^2 + \lambda}\,\mathbf{u}_j\T\by ,
\end{aligned}
$$

其中 $\mathbf{u}_j$ 是 $\mathbf{U}$ 的第 $j$ 列。OLS 以权重 $1$ 把 $\by$ 投影到每个方向 $\mathbf{u}_j$ 上；岭回归则把投影乘以**收缩因子** $d_j^2/(d_j^2+\lambda) \in (0,1)$，对 $d_j$ 较小的方向（即数据所含信息较少的方向）收缩最强。

拟合值为 $\widehat{\by}^{R}_\lambda = \bH_\lambda\by$，其中 $\bH_\lambda = \bX(\bX\T\bX + \lambda\bI_k)^{-1}\bX\T$。类比 OLS 的 $\tr(\bH) = p$，**有效自由度**为

$$
\begin{aligned}
\operatorname{df}(\lambda) &= \tr(\bH_\lambda) \\
&= \sum_{j=1}^k\frac{d_j^2}{d_j^2+\lambda},
\end{aligned}
$$

它从 $\lambda = 0$ 时的 $k$ 随 $\lambda\to\infty$ 递减至 $0$。

## lasso 回归

### lasso 估计量

**lasso**（least absolute shrinkage and selection operator，最小绝对收缩与选择算子；Tibshirani, 1996）求解

$$
\widehat{\bbeta}^{L}_\lambda
= \operatorname*{arg\,min}_{\bbeta\in\R^k}
\Bigl\{\lVert\by - \bX\bbeta\rVert_2^2 + \lambda\lVert\bbeta\rVert_1\Bigr\}.
$$ (eq-lasso)

目标函数是凸的，但在某个 $\beta_j = 0$ 处不可微，因此一般没有闭式解。其解由 {prf:ref}`thm-subgradient-optimality` 的次梯度条件刻画：对每个 $j$，

$$
\begin{gathered}
2\bx_{(j)}\T\bigl(\by - \bX\widehat{\bbeta}^{L}_\lambda\bigr) = \lambda\,\sign\bigl(\widehat{\beta}^{L}_j\bigr) \qquad  \text{若 } \widehat{\beta}^{L}_j \ne 0, \\
\bigl\lvert 2\bx_{(j)}\T\bigl(\by - \bX\widehat{\bbeta}^{L}_\lambda\bigr)\bigr\rvert \le \lambda \qquad  \text{若 } \widehat{\beta}^{L}_j = 0,
\end{gathered}
$$ (eq-lasso-kkt)

其中 $\bx_{(j)}$ 是 $\bX$ 的第 $j$ 列。当某个预测变量与当前残差的相关性太小、不足以抵消惩罚时，它就被排除。特别地，当 $\lambda \ge \lambda_{\max} = 2\max_j\lvert\bx_{(j)}\T\by\rvert$ 时 $\widehat{\bbeta}^{L}_\lambda = \bzero$。

### 正交设计：软阈值

当 $\bX$ 的列标准正交（$\bX\T\bX = \bI_k$）时，三种惩罚估计量都有显式形式，显示出各自如何处理 OLS 估计 $\widehat{\beta}_j = \bx_{(j)}\T\by$。

:::{prf:theorem} 正交设计下的惩罚估计量
:label: thm-orthonormal-penalized

若 $\bX\T\bX = \bI_k$，则对 $j = 1,\ldots,k$，

$$
\begin{aligned}
\text{岭回归：} \quad & \widehat{\beta}^{R}_j = \frac{\widehat{\beta}_j}{1 + \lambda}, \\
\text{lasso：} \quad & \widehat{\beta}^{L}_j = S\bigl(\widehat{\beta}_j,\ \lambda/2\bigr) \\
 &= \sign(\widehat{\beta}_j)\bigl(\lvert\widehat{\beta}_j\rvert - \lambda/2\bigr)_+ , \\
\text{最优子集 } \bigl(P(\bbeta) = \#\{j:\beta_j\ne0\}\bigr): \quad & \widehat{\beta}^{0}_j = \widehat{\beta}_j\,\mathbb{I}\bigl(\lvert\widehat{\beta}_j\rvert > \sqrt{\lambda}\bigr).
\end{aligned}
$$
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

由 $\bX\T\bX = \bI_k$ 和 $\widehat{\bbeta} = \bX\T\by$，

$$
\begin{aligned}
\lVert\by - \bX\bbeta\rVert_2^2
&= \lVert\by - \bX\widehat{\bbeta}\rVert_2^2 + \lVert\bX(\widehat{\bbeta} - \bbeta)\rVert_2^2 \\
&= \lVert\by - \bX\widehat{\bbeta}\rVert_2^2 + \sum_{j=1}^k(\widehat{\beta}_j - \beta_j)^2 ,
\end{aligned}
$$

因为交叉项 $2(\by - \bX\widehat{\bbeta})\T\bX(\widehat{\bbeta}-\bbeta)$ 为零（{prf:ref}`prop-orthogonality`）。因此每个惩罚问题都分解为 $k$ 个一维问题 $\min_{\beta_j}\{(\widehat{\beta}_j - \beta_j)^2 + \lambda P_j(\beta_j)\}$。

- 岭回归：令 $-2(\widehat{\beta}_j - \beta_j) + 2\lambda\beta_j = 0$，得 $\beta_j = \widehat{\beta}_j/(1+\lambda)$。
- lasso：条件 $0 \in -2(\widehat{\beta}_j - \beta_j) + \lambda\,\partial\lvert\beta_j\rvert$ 在 $\beta_j \ne 0$ 时给出 $\beta_j = \widehat{\beta}_j - \frac{\lambda}{2}\sign(\beta_j)$，这只有在 $\lvert\widehat{\beta}_j\rvert > \lambda/2$ 时才自洽；否则 $\lvert\widehat{\beta}_j\rvert \le \lambda/2$，此时 $\beta_j = 0$ 满足 $0 \in [-2\widehat{\beta}_j - \lambda,\ -2\widehat{\beta}_j + \lambda]$。
- 最优子集：保留 $\beta_j = \widehat{\beta}_j$ 的代价为 $\lambda$，令 $\beta_j = 0$ 的代价为 $\widehat{\beta}_j^2$；当 $\widehat{\beta}_j^2 > \lambda$ 时保留该系数。
:::

岭回归把每个系数按相同比例收缩，从不把系数压缩为零。lasso 从每个系数的绝对值中减去常数 $\lambda/2$，并把较小的系数置零（**软阈值**）。最优子集选择保留较大的系数不变，删去较小的系数（**硬阈值**）。

### lasso 为什么产生零

在约束形式中，最小二乘的等值线是以 OLS 估计为中心的椭圆，解是等值线第一次接触约束区域的点。岭回归的约束区域 $\{\bbeta : \lVert\bbeta\rVert_2^2 \le t\}$ 是边界光滑的球，接触点一般所有坐标都不为零。lasso 的约束区域 $\{\bbeta : \lVert\bbeta\rVert_1 \le t\}$ 是一个正轴体（二维时为菱形），其顶点位于坐标轴上，因此接触点常常落在顶点或棱上，使某些坐标精确为零。$\lvert\beta_j\rvert$ 在零处的不可微性（见 {eq}`eq-lasso-kkt`）正是这些顶点的解析对应物。

### 坐标下降

对一般设计，lasso 通常用**循环坐标下降**（coordinate descent）拟合（Friedman, Hastie, & Tibshirani, 2010）。固定除 $\beta_j$ 以外的所有系数，目标函数关于 $\beta_j$ 是一个一维 lasso 问题，其中**偏残差**为 $\mathbf{r}_{(j)} = \by - \sum_{l\ne j}\bx_{(l)}\beta_l$。

:::{prf:algorithm} lasso 的坐标下降
:label: alg-lasso-cd

1. 标准化预测变量，中心化响应，选取初值，例如 $\bbeta = \bzero$。
2. 重复以下步骤直到系数的变化小于容差；对 $j = 1,\ldots,k$：

   $$
   \beta_j \leftarrow \frac{S\bigl(\bx_{(j)}\T\mathbf{r}_{(j)},\ \lambda/2\bigr)}{\bx_{(j)}\T\bx_{(j)}} .
   $$
3. 恢复截距 $\widehat{\beta}_0 = \bar{y}$，并把系数换算回原始尺度。
:::

该更新由把正交设计结论用于单列得到：最小化 $\lVert\mathbf{r}_{(j)} - \bx_{(j)}\beta_j\rVert_2^2 + \lambda\lvert\beta_j\rvert$ 给出 $\beta_j\,\bx_{(j)}\T\bx_{(j)} = S(\bx_{(j)}\T\mathbf{r}_{(j)}, \lambda/2)$。对从 $\lambda_{\max}$ 开始递减的一列 $\lambda$ 值依次求解，并以每个解作为下一个问题的初值，就得到**正则化路径**。

### 弹性网

当预测变量高度相关时，lasso 往往较随意地只选其中一个。**弹性网**（elastic net；Zou & Hastie, 2005）加入岭回归项，

$$
\min_{\bbeta}\Bigl\{\lVert\by - \bX\bbeta\rVert_2^2 + \lambda_1\lVert\bbeta\rVert_1 + \lambda_2\lVert\bbeta\rVert_2^2\Bigr\},
$$

它保留了 lasso 的稀疏性，同时让相关的预测变量共享权重。

## 贝叶斯解释

设 $\by\mid\bbeta \sim \mathcal{N}(\bX\bbeta, \sigma^2\bI_n)$。在先验密度 $\pi(\bbeta)$ 下，负对数后验（相差一个常数）为 $\lVert\by - \bX\bbeta\rVert_2^2/(2\sigma^2) - \ln\pi(\bbeta)$。

- 取独立正态先验 $\beta_j \sim \mathcal{N}(0,\tau^2)$，后验众数（也是后验均值）就是 $\lambda = \sigma^2/\tau^2$ 的岭估计量。
- 取独立 Laplace 先验 $\beta_j \sim \mathrm{Laplace}(0,b)$，后验众数就是 $\lambda = 2\sigma^2/b$ 的 lasso 估计量。

Laplace 先验在零处有尖峰，这是 lasso 偏好精确零值的概率解释。

## 选择惩罚参数

惩罚参数 $\lambda$ 应使预测误差的估计最小，最常用的方法是 **$K$ 折交叉验证**（cross-validation）：

1. 把数据分成 $K$ 折（常取 $K = 5$ 或 $10$）；
2. 对每个候选 $\lambda$ 和每一折 $m$，用其余各折拟合模型，计算在第 $m$ 折上的平方预测误差；
3. 对各折取平均得到 $\mathrm{CV}(\lambda)$，选择使其最小的 $\lambda$。

**一倍标准误规则**选择 $\mathrm{CV}(\lambda)$ 不超过最小值加一个标准误的最大 $\lambda$，从而得到精度相当而更简单的模型。对岭回归，留一交叉验证有类似于 PRESS（{doc}`13-prediction-and-model-assessment`）的闭式：

$$
\mathrm{CV}_{\mathrm{LOO}}(\lambda)
= \frac1n\sum_{i=1}^n\left(\frac{y_i - \widehat{y}_{i,\lambda}}{1 - h_{ii}(\lambda)}\right)^2,
$$

其中 $\widehat{y}_{i,\lambda} = \bar{y} + \bx_i\T\widehat{\bbeta}^{R}_\lambda$ 为含截距的拟合值，$h_{ii}(\lambda)$ 是完整岭回归帽子矩阵 $\bH_\lambda + \frac1n\mathbf{1}_n\mathbf{1}_n\T$ 的第 $i$ 个对角元，其中第二项对应于估计的截距。也可以用 AIC、BIC 等信息准则，以 $\operatorname{df}(\lambda)$ 代替 $p$；对 lasso，非零系数的个数是自由度的无偏估计（Zou, Hastie, & Tibshirani, 2007）。

## 惩罚之后的推断

岭估计量和 lasso 估计量都是有偏的，因此 {doc}`10-anova-and-inference` 中的 $t$ 检验和 $F$ 检验不能直接用于它们。对 lasso，被选中的预测变量集合本身是随机的；把所选模型当作事先设定的模型来计算普通 $p$ 值，会夸大显著性。因此惩罚回归主要用于预测和变量筛选；正式的推断需要专门的选择后推断方法。

## 本章回顾
### 小结

- 惩罚最小二乘在标准化的预测变量上最小化 $\lVert\by-\bX\bbeta\rVert_2^2 + \lambda P(\bbeta)$，截距不受惩罚。
- 岭回归：$\widehat{\bbeta}^R_\lambda = (\bX\T\bX + \lambda\bI)^{-1}\bX\T\by$，偏差为 $-\lambda(\bX\T\bX + \lambda\bI)^{-1}\bbeta$，但方差更小；总存在某个 $\lambda > 0$ 使其均方误差小于 OLS。
- 岭回归把方向 $j$ 收缩 $d_j^2/(d_j^2+\lambda)$ 倍；$\operatorname{df}(\lambda) = \sum_j d_j^2/(d_j^2+\lambda)$。
- lasso：没有闭式解；正交设计下为软阈值；一般情形用坐标下降；产生稀疏解。
- 用交叉验证选择 $\lambda$，可结合一倍标准误规则。

### 常见错误

- 对未标准化的预测变量施加惩罚，使结果依赖于度量单位。
- 对截距施加惩罚。
- 未检查目标函数的缩放方式就在不同软件之间比较 $\lambda$；例如 `scikit-learn` 的 `Lasso` 把平方误差除以 $2n$，而 `Ridge` 不除（{doc}`26-penalized-regression-python`）。
- 把被 lasso 置零的系数当作该预测变量没有作用的证据。
- 为惩罚估计报告 OLS 式的标准误和 $p$ 值。

## 参考文献

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction* (2nd ed.). Springer. <https://doi.org/10.1007/978-0-387-84858-7>
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An introduction to statistical learning: With applications in R*. Springer. <https://doi.org/10.1007/978-1-4614-7138-7>
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software, 33*(1), 1–22. <https://doi.org/10.18637/jss.v033.i01>
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics, 12*(1), 55–67. <https://doi.org/10.1080/00401706.1970.10488634>
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B (Methodological), 58*(1), 267–288. <https://doi.org/10.1111/j.2517-6161.1996.tb02080.x>
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B (Statistical Methodology), 67*(2), 301–320. <https://doi.org/10.1111/j.1467-9868.2005.00503.x>
- Zou, H., Hastie, T., & Tibshirani, R. (2007). On the "degrees of freedom" of the lasso. *The Annals of Statistics, 35*(5), 2173–2192. <https://doi.org/10.1214/009053607000000127>

# 概率与随机向量

## 概述

回归模型描述的是给定一组预测变量时响应变量的分布，因此后续每一章都要用到概率论的语言。本章回顾随机变量、期望与方差、条件期望、书中用到的各种分布，以及随机向量的均值向量和协方差矩阵。最后介绍多元正态分布，它是线性模型精确推断的基础。 本章假设读者已掌握微积分和基本的矩阵记号（本章用到的矩阵结论见 {doc}`05-linear-algebra`）。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 计算期望、方差和协方差，并运用它们的线性运算法则；
- 运用全期望公式和全方差公式；
- 写出书中所用各分布的均值和方差；
- 计算随机向量线性变换的均值向量与协方差矩阵，以及二次型的期望；
- 陈述多元正态分布的主要性质。
:::

## 随机变量与分布

**随机变量**（random variable）$X$ 是随机试验的数值结果。其**累积分布函数**（cumulative distribution function, CDF）为 $F(x) = \Prob(X \le x)$，它单调不减、右连续，且 $F(-\infty) = 0$，$F(\infty) = 1$。

- **离散型**随机变量取可数个值，具有**概率质量函数**（probability mass function, PMF）$f(x) = \Prob(X = x)$。
- **连续型**随机变量具有**概率密度函数**（probability density function, PDF）$f(x) \ge 0$，且 $F(x) = \int_{-\infty}^{x} f(t)\,dt$。

在需要区分时，本书用大写字母（如 $Y_i$）表示随机变量，用小写字母（如 $y_i$）表示其观测值。

## 期望与方差

### 期望

$X$ 的**期望**（expectation，即均值）为

$$
\begin{gathered}
\E[X] = \sum_x x\,f(x) \quad\text{（离散）}, \\
\E[X] = \int_{-\infty}^{\infty} x\,f(x)\,dx \quad\text{（连续）},
\end{gathered}
$$

对函数 $g$，$\E[g(X)] = \sum_x g(x)f(x)$ 或 $\int g(x)f(x)\,dx$。期望是**线性**的：对常数 $a$、$b$ 和随机变量 $X$、$Y$，

$$
\E[aX + bY] = a\,\E[X] + b\,\E[Y],
$$

无论 $X$ 与 $Y$ 是否独立，上式都成立。

### 方差、协方差与相关系数

**方差**（variance）衡量 $X$ 围绕其均值的离散程度：

$$
\begin{aligned}
\Var(X) &= \E\bigl[(X - \E[X])^2\bigr] \\
        &= \E[X^2] - \bigl(\E[X]\bigr)^2 .
\end{aligned}
$$

$X$ 与 $Y$ 的**协方差**（covariance）和**相关系数**（correlation）为

$$
\begin{aligned}
\Cov(X,Y) &= \E\bigl[(X-\E[X])(Y-\E[Y])\bigr] \\
          &= \E[XY] - \E[X]\,\E[Y], \\
\operatorname{Corr}(X,Y) &= \frac{\Cov(X,Y)}{\sqrt{\Var(X)\Var(Y)}} \in [-1, 1].
\end{aligned}
$$

对常数 $a$、$b$、$c$，

$$
\begin{gathered}
\Var(aX + c) = a^2\Var(X), \\
\Var(aX + bY) = a^2\Var(X) + b^2\Var(Y) + 2ab\,\Cov(X,Y).
\end{gathered}
$$

若 $X$ 与 $Y$ 独立，则 $\E[XY] = \E[X]\E[Y]$，$\Cov(X,Y) = 0$。反之一般不成立：不相关的随机变量不一定独立；但对联合正态变量，二者等价（{prf:ref}`thm-mvn-properties`）。

### 条件期望

**条件期望**（conditional expectation）$\E[Y\mid X]$ 是由给定 $X$ 时 $Y$ 的条件分布计算的均值，它是 $X$ 的函数。回归模型就是关于 $\E[Y\mid\bx]$ 的模型。

:::{prf:theorem} 全期望公式与全方差公式
:label: thm-total-expectation

对方差有限的随机变量 $X$ 和 $Y$，

$$
\begin{gathered}
\E[Y] = \E\bigl[\E[Y\mid X]\bigr], \\
\Var(Y) = \E\bigl[\Var(Y\mid X)\bigr] + \Var\bigl(\E[Y\mid X]\bigr).
\end{gathered}
$$
:::

第二个等式把 $Y$ 的方差分解为围绕条件均值的平均波动与条件均值本身的波动两部分。例如，它解释了均值随机变化的泊松计数为何会过度离散（{doc}`19-poisson-and-count-regression`）。

## 常用分布

下表列出书中用到的分布及其 $\E[X]$ 和 $\Var(X)$。

### 离散分布

| 分布 | PMF $f(x)$ | 取值范围 | $\E[X]$ | $\Var(X)$ |
|---|---|---|---|---|
| $\mathrm{Bernoulli}(\pi)$ | $\pi^x(1-\pi)^{1-x}$ | $\{0,1\}$ | $\pi$ | $\pi(1-\pi)$ |
| $\mathrm{Binomial}(m,\pi)$ | $\binom{m}{x}\pi^x(1-\pi)^{m-x}$ | $\{0,\ldots,m\}$ | $m\pi$ | $m\pi(1-\pi)$ |
| $\mathrm{Poisson}(\lambda)$ | $e^{-\lambda}\lambda^x/x!$ | $\{0,1,2,\ldots\}$ | $\lambda$ | $\lambda$ |
| $\mathrm{NegBin}(r,p_s)$ | $\binom{x+r-1}{r-1}(1-p_s)^x p_s^{\,r}$ | $\{0,1,2,\ldots\}$ | $r(1-p_s)/p_s$ | $r(1-p_s)/p_s^2$ |

### 连续分布

| 分布 | PDF $f(x)$ | 取值范围 | $\E[X]$ | $\Var(X)$ |
|---|---|---|---|---|
| $\mathrm{Uniform}(a,b)$ | $1/(b-a)$ | $[a,b]$ | $(a+b)/2$ | $(b-a)^2/12$ |
| $\mathcal{N}(\mu,\sigma^2)$ | $\dfrac{1}{\sqrt{2\pi}\,\sigma}e^{-(x-\mu)^2/(2\sigma^2)}$ | $\R$ | $\mu$ | $\sigma^2$ |
| $\mathrm{Exponential}(\lambda)$ | $\lambda e^{-\lambda x}$ | $[0,\infty)$ | $1/\lambda$ | $1/\lambda^2$ |
| $\mathrm{Gamma}(\alpha,\theta)$（形状、尺度） | $\dfrac{x^{\alpha-1}e^{-x/\theta}}{\Gamma(\alpha)\theta^\alpha}$ | $(0,\infty)$ | $\alpha\theta$ | $\alpha\theta^2$ |
| $\mathrm{Laplace}(\mu,b)$ | $\dfrac{1}{2b}e^{-\lvert x-\mu\rvert/b}$ | $\R$ | $\mu$ | $2b^2$ |
| $\mathrm{Logistic}(\mu,s)$ | $\dfrac{e^{-(x-\mu)/s}}{s\bigl(1+e^{-(x-\mu)/s}\bigr)^2}$ | $\R$ | $\mu$ | $\pi^2s^2/3$ |
| $\mathrm{Gumbel}(\eta,\mu)$（位置参数 $\eta$；$\mu > 0$，$\mu$ 越大分布越集中） | $\mu\,e^{-z}e^{-e^{-z}}$，$z = \mu(x-\eta)$ | $\R$ | $\eta + \gamma/\mu$ | $\pi^2/(6\mu^2)$ |

其中 $\Gamma(\cdot)$ 为伽马函数，$\gamma \approx 0.57721$ 为 Euler–Mascheroni 常数。标准正态分布 $\mathcal{N}(0,1)$ 的分布函数记为 $\Phi$，密度函数记为 $\phi$。logistic 分布函数为 $F(x) = 1/\bigl(1 + e^{-(x-\mu)/s}\bigr)$；当 $\mu = 0$、$s = 1$ 时，它就是 {doc}`18-logistic-regression` 中 logit 函数的反函数。Gumbel 分布用于 {doc}`21-multinomial-choice-models`，Laplace 分布用于 {doc}`16-ridge-and-lasso`。

## 随机向量

**随机向量**（random vector）$\bz = (Z_1,\ldots,Z_m)\T$ 是由随机变量组成的向量。其**均值向量**和**协方差矩阵**为

$$
\E[\bz] = \bigl(\E[Z_1],\ldots,\E[Z_m]\bigr)\T,
$$

$$
\begin{aligned}
\Cov(\bz) &= \E\bigl[(\bz - \E[\bz])(\bz - \E[\bz])\T\bigr] \\
          &= \bigl[\Cov(Z_j, Z_l)\bigr]_{j,l=1}^{m} .
\end{aligned}
$$

协方差矩阵对称且半正定，其对角元为方差 $\Var(Z_j)$，也记作 $\Var(\bz)$。随机向量 $\bz$ 与 $\mathbf{w}$ 的**互协方差**为 $\Cov(\bz,\mathbf{w}) = \E\bigl[(\bz-\E[\bz])(\mathbf{w}-\E[\mathbf{w}])\T\bigr]$。

:::{prf:lemma} 随机向量的线性变换
:label: lem-linear-transform

设 $\mathbf{A}$、$\mathbf{B}$ 为常数（非随机）矩阵，$\mathbf{c}$ 为常数向量，$\bz$ 为维数相容的随机向量，则

$$
\begin{gathered}
\E[\mathbf{A}\bz + \mathbf{c}] = \mathbf{A}\,\E[\bz] + \mathbf{c}, \\
\Cov(\mathbf{A}\bz + \mathbf{c}) = \mathbf{A}\,\Cov(\bz)\,\mathbf{A}\T, \\
\Cov(\mathbf{A}\bz,\ \mathbf{B}\bz) = \mathbf{A}\,\Cov(\bz)\,\mathbf{B}\T .
\end{gathered}
$$
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

第一个等式由期望的线性性逐分量得到。对第二个等式，记 $\boldsymbol\mu = \E[\bz]$，则 $\mathbf{A}\bz + \mathbf{c} - \E[\mathbf{A}\bz + \mathbf{c}] = \mathbf{A}(\bz - \boldsymbol\mu)$，于是

$$
\begin{aligned}
\Cov(\mathbf{A}\bz + \mathbf{c})
&= \E\bigl[\mathbf{A}(\bz-\boldsymbol\mu)(\bz-\boldsymbol\mu)\T\mathbf{A}\T\bigr] \\
&= \mathbf{A}\,\E\bigl[(\bz-\boldsymbol\mu)(\bz-\boldsymbol\mu)\T\bigr]\,\mathbf{A}\T \\
&= \mathbf{A}\,\Cov(\bz)\,\mathbf{A}\T .
\end{aligned}
$$

第三个等式的证明类似。
:::

:::{prf:lemma} 二次型的期望
:label: lem-quadratic-form-mean

设随机向量 $\bz$ 的均值为 $\boldsymbol\mu$、协方差矩阵为 $\bSigma$，$\mathbf{A}$ 为常数对称矩阵，则

$$
\E\bigl[\bz\T\mathbf{A}\bz\bigr] = \tr(\mathbf{A}\bSigma) + \boldsymbol\mu\T\mathbf{A}\boldsymbol\mu .
$$
:::

:::{dropdown} 证明
:class-container: proof
:animate: fade-in

标量等于其自身的迹，而迹是线性的且具有循环性质（{prf:ref}`lem-trace-cyclic`），因此

$$
\begin{aligned}
\E\bigl[\bz\T\mathbf{A}\bz\bigr]
&= \E\bigl[\tr(\mathbf{A}\bz\bz\T)\bigr] \\
 &= \tr\bigl(\mathbf{A}\,\E[\bz\bz\T]\bigr) \\
&= \tr\bigl(\mathbf{A}(\bSigma + \boldsymbol\mu\boldsymbol\mu\T)\bigr) \\
 &= \tr(\mathbf{A}\bSigma) + \boldsymbol\mu\T\mathbf{A}\boldsymbol\mu .
\end{aligned}
$$
:::

利用这一引理，可以在 {doc}`09-ols-properties-and-geometry` 中一行证明 $\E[\mathit{SS}_E] = (n-p)\sigma^2$。

## 多元正态分布

若随机向量 $\bz\in\R^m$ 的密度为

$$
f(\bz) = (2\pi)^{-m/2}\,\lvert\bSigma\rvert^{-1/2}
\exp\Bigl[-\tfrac12(\bz - \boldsymbol\mu)\T\bSigma^{-1}(\bz - \boldsymbol\mu)\Bigr],
$$

其中 $\lvert\bSigma\rvert$ 为 $\bSigma$ 的行列式，则称 $\bz$ 服从均值为 $\boldsymbol\mu$、协方差矩阵为正定矩阵 $\bSigma$ 的**多元正态分布**（multivariate normal distribution），记为 $\bz \sim \mathcal{N}_m(\boldsymbol\mu, \bSigma)$。标准情形 $\mathcal{N}_m(\bzero, \bI_m)$ 是由 $m$ 个独立的 $\mathcal{N}(0,1)$ 变量组成的向量。

:::{prf:theorem} 多元正态分布的性质
:label: thm-mvn-properties

设 $\bz \sim \mathcal{N}_m(\boldsymbol\mu, \bSigma)$。

1. **线性变换。** 对常数 $q\times m$ 矩阵 $\mathbf{A}$ 和向量 $\mathbf{c}$，$\mathbf{A}\bz + \mathbf{c} \sim \mathcal{N}_q(\mathbf{A}\boldsymbol\mu + \mathbf{c},\ \mathbf{A}\bSigma\mathbf{A}\T)$。
2. **边缘分布。** $\bz$ 的任何子向量都服从多元正态分布；特别地 $Z_j \sim \mathcal{N}(\mu_j, \Sigma_{jj})$。
3. **独立性。** 若将 $\bz$ 划分为 $(\bz_1\T, \bz_2\T)\T$，则 $\bz_1$ 与 $\bz_2$ 独立当且仅当 $\Cov(\bz_1, \bz_2) = \mathbf{0}$。更一般地，$\mathbf{A}\bz$ 与 $\mathbf{B}\bz$ 独立当且仅当 $\mathbf{A}\bSigma\mathbf{B}\T = \mathbf{0}$。
:::

性质 3 是正态分布特有的：协方差为零即意味着独立。这是证明正态线性模型中 $\widehat{\bbeta}$ 与残差相互独立的关键一步（{prf:ref}`thm-ols-distribution`）。

## 本章回顾
### 小结

- 期望是线性的；$\Var(aX + bY) = a^2\Var(X) + b^2\Var(Y) + 2ab\Cov(X,Y)$。
- $\E[Y] = \E[\E[Y\mid X]]$，$\Var(Y) = \E[\Var(Y\mid X)] + \Var(\E[Y\mid X])$。
- $\E[\mathbf{A}\bz] = \mathbf{A}\E[\bz]$，$\Cov(\mathbf{A}\bz) = \mathbf{A}\Cov(\bz)\mathbf{A}\T$，$\E[\bz\T\mathbf{A}\bz] = \tr(\mathbf{A}\bSigma) + \boldsymbol\mu\T\mathbf{A}\boldsymbol\mu$。
- 多元正态向量的线性变换仍服从正态分布；对联合正态向量，协方差为零即独立。

### 常见错误

- 未检查 $X$ 与 $Y$ 是否不相关，就使用 $\Var(X + Y) = \Var(X) + \Var(Y)$。
- 对非正态变量，由协方差为零推出独立。
- 写成 $\Cov(\mathbf{A}\bz) = \mathbf{A}\T\Cov(\bz)\mathbf{A}$；转置应在右边。

## 参考文献

- Casella, G., & Berger, R. L. (2002). *Statistical inference* (2nd ed.). Duxbury.
- Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley.
- Seber, G. A. F., & Lee, A. J. (2003). *Linear regression analysis* (2nd ed.). Wiley. <https://doi.org/10.1002/9780471722199>

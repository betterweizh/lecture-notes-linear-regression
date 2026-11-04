# 预测与模型评价

## 概述

拟合好的回归模型可以用来估计平均响应、预测新的观测值，但在相信其结论之前必须先对模型进行检验。本章推导置信区间和预测区间，介绍拟合优度指标，并给出残差诊断和影响诊断方法。 本章以帽子矩阵和抽样分布（{doc}`09-ols-properties-and-geometry`）以及平方和与 $t$ 统计量（{doc}`10-anova-and-inference`）为基础。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 构造平均响应的置信区间和新观测值的预测区间，并解释后者为何更宽；
- 计算并解释 $R^2$、调整 $R^2$、PRESS 和预测 $R^2$；
- 计算标准化残差、学生化残差和杠杆值；
- 用 Cook 距离识别强影响点。
:::

## 在新点处的预测

设 $\bx_0 = (1, x_{01}, \ldots, x_{0k})\T \in \R^p$ 为新观测的预测变量取值，其响应为

$$
\begin{gathered}
y_0 = \bx_0\T\bbeta + \varepsilon_0, \\
\varepsilon_0 \sim \mathcal{N}(0,\sigma^2),
\end{gathered}
$$

其中 $\varepsilon_0$ 与用于拟合模型的数据中的误差独立。平均响应 $\mu_0 = \E[y_0] = \bx_0\T\bbeta$ 与新响应 $y_0$ 的点估计都是拟合值

$$
\widehat{y}_0 = \bx_0\T\widehat{\bbeta}
\ \sim\ \mathcal{N}\bigl(\bx_0\T\bbeta,\ \sigma^2\,\bx_0\T(\bX\T\bX)^{-1}\bx_0\bigr),
$$

这由 {prf:ref}`thm-ols-distribution` 和 {prf:ref}`lem-linear-transform` 得到。特别地，$\widehat{y}_0$ 是 $\mu_0$ 的无偏估计量。

### 平均响应的置信区间

用 $\widehat{\sigma}$ 代替 $\sigma$，并利用 $\widehat{\bbeta}$ 与 $\widehat{\sigma}^2$ 的独立性，

$$
\frac{\widehat{y}_0 - \bx_0\T\bbeta}{\widehat{\sigma}\sqrt{\bx_0\T(\bX\T\bX)^{-1}\bx_0}} \ \sim\ t(n-p).
$$

因此 $\mu_0$ 的 $100(1-\alpha)\%$ **置信区间**（confidence interval, CI）为

$$
\widehat{y}_0 \pm t_{1-\alpha/2,\,n-p}\ \widehat{\sigma}\sqrt{\bx_0\T(\bX\T\bX)^{-1}\bx_0}.
$$ (eq-ci-mean)

### 新观测值的预测区间

$y_0$ 的**预测区间**（prediction interval, PI）$[l, u]$ 满足 $\Prob(l \le y_0 \le u) = 1-\alpha$。预测误差为

$$
y_0 - \widehat{y}_0
= \varepsilon_0 - \bx_0\T(\widehat{\bbeta}-\bbeta)
\ \sim\ \mathcal{N}\bigl(0,\ \sigma^2\bigl[1 + \bx_0\T(\bX\T\bX)^{-1}\bx_0\bigr]\bigr),
$$

这是因为新误差 $\varepsilon_0$ 与 $\widehat{\bbeta}$ 独立，两部分方差相加。用 $\widehat{\sigma}$ 标准化得

$$
\frac{y_0 - \widehat{y}_0}{\widehat{\sigma}\sqrt{1 + \bx_0\T(\bX\T\bX)^{-1}\bx_0}} \ \sim\ t(n-p),
$$

于是 $100(1-\alpha)\%$ 预测区间为

$$
\widehat{y}_0 \pm t_{1-\alpha/2,\,n-p}\ \widehat{\sigma}\sqrt{1 + \bx_0\T(\bX\T\bX)^{-1}\bx_0}.
$$ (eq-pi)

:::{prf:remark}
:label: rem-pi-wider

在同一 $\bx_0$ 处，预测区间 {eq}`eq-pi` 总是比置信区间 {eq}`eq-ci-mean` 宽：置信区间只考虑 $\widehat{\bbeta}$ 的不确定性，而预测区间还包含单个新观测围绕其均值的波动 $\sigma^2$。当 $n\to\infty$ 时，置信区间收缩为一点，而预测区间不会。
:::

## 拟合优度

### 决定系数

**（复）决定系数**（coefficient of determination）为

$$
\begin{aligned}
R^2 &= \frac{\mathit{SS}_R}{\mathit{SS}_T} \\
&= 1 - \frac{\mathit{SS}_E}{\mathit{SS}_T}.
\end{aligned}
$$

对含截距的模型：

- $R^2$ 是 $y$ 的观测变异中可由预测变量 $x_1,\ldots,x_k$ 解释的比例；
- $0 \le R^2 \le 1$，当所有残差为零时 $R^2 = 1$；
- 增加一个预测变量，无论其是否有用，$R^2$ 都不会减小，因此不能单独用 $R^2$ 比较不同规模的模型。

### 调整 $R^2$

**调整 $R^2$**（adjusted $R^2$）用均方代替平方和，从而对增加的系数进行惩罚：

$$
\begin{aligned}
R^2_{\mathrm{adj}} &= 1 - \frac{\mathit{SS}_E/(n-p)}{\mathit{SS}_T/(n-1)} \\
&= 1 - \frac{n-1}{n-p}\bigl(1-R^2\bigr) \\
&= \frac{(n-1)R^2 - k}{n-k-1}.
\end{aligned}
$$

性质：

- $R^2_{\mathrm{adj}} \le R^2 \le 1$；与 $R^2$ 不同，当模型拟合很差时调整 $R^2$ 可以为负；
- 只有当 $\mathit{SS}_E$ 的减少足以弥补损失的一个残差自由度时，加入预测变量才会使 $R^2_{\mathrm{adj}}$ 增大；
- 若 $R^2$ 与 $R^2_{\mathrm{adj}}$ 相差很大，通常说明模型中包含贡献很小的预测变量。

### PRESS 与预测 $R^2$

留一交叉验证（leave-one-out cross-validation, LOOCV）将模型重新拟合 $n$ 次，每次留出一个观测。记 $\widehat{y}_{(i)}$ 为不使用观测 $i$ 拟合的模型对 $y_i$ 的预测。**预测误差平方和**（PRESS）为

$$
\begin{aligned}
\mathrm{PRESS} &= \sum_{i=1}^n \bigl(y_i - \widehat{y}_{(i)}\bigr)^2 \\
&= \sum_{i=1}^n \left(\frac{e_i}{1-h_{ii}}\right)^2,
\end{aligned}
$$

第二种形式利用下文定义的杠杆值 $h_{ii}$，无需重新拟合。**预测 $R^2$** 为

$$
\begin{gathered}
R^2_{\mathrm{pred}} = 1 - \frac{\mathrm{PRESS}}{\mathit{SS}_T}, \\
\mathit{SS}_T = \sum_{i=1}^n (y_i-\bar{y})^2 .
\end{gathered}
$$

它衡量模型对未参与拟合的观测的预测能力。

## 残差分析

### 杠杆值

帽子矩阵 {eq}`eq-hat-matrix` 的对角元

$$
h_{ii} = \bx_i\T(\bX\T\bX)^{-1}\bx_i, \qquad i = 1,\ldots,n,
$$

称为**杠杆值**（leverage），其中 $\bx_i\T = (1, x_{i1},\ldots,x_{ik})$ 是 $\bX$ 的第 $i$ 行。它们满足 $0 \le h_{ii} \le 1$ 且 $\sum_i h_{ii} = \tr(\bH) = p$。$h_{ii}$ 较大说明观测 $i$ 的预测变量取值远离数据中心。

### 残差的类型

由于 $\be = (\bI_n-\bH)\by$ 且 $\bI_n-\bH$ 对称幂等，$\Cov(\be) = \sigma^2(\bI_n-\bH)$，从而 $\Var(e_i) = \sigma^2(1-h_{ii})$。由此有三种尺度的残差。

普通残差
: $e_i = y_i - \widehat{y}_i$。

标准化残差
: $d_i = \dfrac{e_i}{\widehat{\sigma}}$；模型正确且 $n$ 较大时近似服从 $\mathcal{N}(0,1)$。

学生化残差
: $r_i = \dfrac{e_i}{\sqrt{\widehat{\sigma}^2(1-h_{ii})}}$，即用每个残差标准差的估计值去除，也称内学生化残差。

由于 $0 < 1-h_{ii} \le 1$，有 $|r_i| \ge |d_i|$。学生化残差修正了高杠杆点处残差方差偏小的问题，因而更适合检测离群点；$|r_i| > 3$ 的观测值值得关注。

### 残差图

将残差分别对拟合值和各预测变量作图。若呈现无规律的水平带状，则支持模型；若呈弯曲形态，说明可能遗漏了非线性项；若呈喇叭形，说明方差可能不恒定。残差的正态分位数–分位数图（Q–Q 图）用于检验正态性假设（{doc}`04-goodness-of-fit-tests`）。

## 强影响点

**强影响点**（influential observation）是指删除后会使拟合模型发生明显变化的观测：参数估计或预测更多地依赖于少数观测而不是大多数数据。识别这些观测很重要，原因有二：

- 若它们是错误数据，应当修正或删除；
- 若它们是有效数据，它们决定了模型的关键性质，分析者应当知道这一点。

### Cook 距离

Cook 距离（Cook, 1977）比较包含与不包含观测 $i$ 时的拟合值。记 $\widehat{\bbeta}_{(i)}$ 为删除观测 $i$ 后的 OLS 估计量，$\widehat{\by}_{(i)} = \bX\widehat{\bbeta}_{(i)}$。两个拟合向量之间的欧氏距离平方为

$$
\lVert\widehat{\by}_{(i)}-\widehat{\by}\rVert_2^2
= \bigl(\widehat{\bbeta}_{(i)}-\widehat{\bbeta}\bigr)\T\bX\T\bX\bigl(\widehat{\bbeta}_{(i)}-\widehat{\bbeta}\bigr).
$$

除以 $p\,\widehat{\sigma}^2$，得到 **Cook 距离**

$$
\begin{aligned}
D_i &= \frac{\bigl(\widehat{\bbeta}_{(i)}-\widehat{\bbeta}\bigr)\T\bX\T\bX\bigl(\widehat{\bbeta}_{(i)}-\widehat{\bbeta}\bigr)}{p\,\widehat{\sigma}^2} \\
&= \frac{r_i^2}{p}\cdot\frac{h_{ii}}{1-h_{ii}}, \qquad i = 1,\ldots,n,
\end{aligned}
$$ (eq-cooks-distance)

其中 $r_i$ 为学生化残差。第二种形式表明 $D_i$ 由两部分组成：$r_i^2$ 反映模型对观测 $i$ 拟合得有多差；$h_{ii}/(1-h_{ii})$ 衡量该点在预测变量空间中离其余数据有多远。

常用的经验规则是：当 $D_i > 1$ 时，将观测 $i$ 视为强影响点。

## 本章回顾
### 小结

- 平均响应置信区间的半宽为 $t_{1-\alpha/2,n-p}\,\widehat{\sigma}\sqrt{\bx_0\T(\bX\T\bX)^{-1}\bx_0}$；预测区间在根号内多加 $1$。
- $R^2$ 随预测变量增多而不减；$R^2_{\mathrm{adj}}$ 和 $R^2_{\mathrm{pred}}$ 会惩罚无用的预测变量。
- $\Var(e_i) = \sigma^2(1-h_{ii})$；学生化残差考虑了杠杆值。
- Cook 距离综合了残差大小和杠杆值；$D_i > 1$ 是常用的警示阈值。

### 常见错误

- 问题涉及单个未来观测时却使用置信区间；这时需要更宽的预测区间。
- 用 $R^2$ 选择模型；它总是偏向更大的模型。
- 不考虑杠杆值，就把较大的普通残差当作离群点。
- 自动删除强影响点。影响大是需要调查的理由，而不是删除的理由。

## 参考文献

- Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley.
- Cook, R. D. (1977). Detection of influential observation in linear regression. *Technometrics, 19*(1), 15–18. <https://doi.org/10.1080/00401706.1977.10489493>
- He, Z. [何志坚]. (n.d.). *数理统计讲义* [Lecture notes on mathematical statistics]. <https://bookdown.org/hezhijian/book/>

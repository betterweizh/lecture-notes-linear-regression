# 用 R 实现线性回归

## 概述

本章用 R 的基础函数 `lm()` 实现线性回归，内容包括：模型公式（增删预测变量、截距、变换、分类预测变量和交互项），残差、拟合值与区间估计，用 `alias()` 和 `vif()` 检查多重共线性，以及 {doc}`15-variable-selection` 中的子集选择和逐步选择方法。 本章应用 {doc}`07-linear-regression-model`、{doc}`13-prediction-and-model-assessment`、{doc}`14-standardization-and-multicollinearity` 和 {doc}`15-variable-selection` 中的方法。代码片段中，`data` 是一个包含响应 `Y` 和预测变量 `X1`、`X2`、… 的数据框。基础 R 代码和 `vif()` 已在 R 4.5 下运行；`leaps` 包的调用（`regsubsets()` 及其 `summary`、`plot`、`coef` 和 `vcov` 方法）已对照程序包源代码检查，但未实际运行。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 用 R 公式设定线性模型，包括变换、因子和交互项；
- 提取残差和拟合值，并计算置信区间和预测区间；
- 用 `alias()` 发现完全共线性，用 `car::vif()` 计算 VIF；
- 用 `regsubsets()` 和 `step()` 进行最优子集选择和逐步选择。
:::

## 模型公式

### 选择预测变量与截距

```r
model <- lm(Y ~ X1 + X2, data = data)          # selected predictors
model <- lm(Y ~ ., data = data)                # all other columns as predictors
model <- lm(Y ~ . - X1 - X2, data = data)      # all except X1 and X2

model <- lm(Y ~ 1 + X1 + X2, data = data)      # with intercept (the default)
model <- lm(Y ~ X1 + X2 - 1, data = data)      # without intercept
model <- lm(Y ~ 0 + X1 + X2, data = data)      # without intercept (same model)
model <- lm(Y ~ 1, data = data)                # intercept only
```

请通过 `data` 参数传入数据框，而不要调用 `attach(data)`；后者使变量全局可见，可能悄悄用到其他地方的同名变量。

### 变换

在公式中，`^`、`*`、`+` 和 `-` 有特殊含义，因此对变量做算术运算时必须用 `I()` 保护：

```r
# Y on X1, X1^2, X1^3 (a cubic polynomial in X1)
model <- lm(Y ~ X1 + I(X1^2) + I(X1^3), data = data)
model <- lm(Y ~ log(X2), data = data)          # functions need no I()
```

### 分类预测变量

有 $L$ 个类别的分类预测变量用 $L-1$ 个示性（虚拟）变量表示：

$$
x_{(l)} =
\begin{cases}
1, & \text{若观测属于类别 } l, \\
0, & \text{其他，}
\end{cases}
\qquad l = 1,\ldots,L-1 .
$$

没有示性变量的类别称为**参照类别**（reference category）；每个系数都是某一类别与参照类别的比较。对于因子（以及字符变量），R 会自动生成示性变量：

```r
# suppose X2 takes the values 1, 2, 3
data$X2_2 <- as.numeric(data$X2 == 2)
data$X2_3 <- as.numeric(data$X2 == 3)
model1 <- lm(Y ~ X1 + X2_2 + X2_3, data = data)     # manual indicators
model2 <- lm(Y ~ X1 + factor(X2), data = data)      # same model
```

### 交互项

- **连续–连续。** 对连续变量 $x_1$ 和 $x_2$，交互项是逐元素（Hadamard）乘积 $x_1 \odot x_2$，增加一个新的预测变量。
- **分类–连续。** 对有 $L$ 个类别的分类变量 $x_1$ 和连续变量 $x_2$，交互项增加 $L-1$ 个预测变量，即 $x_2$ 与 $x_1$ 各示性变量的乘积；于是每个类别在 $x_2$ 上都有自己的斜率。
- **分类–分类。** 对分别有 $L_1$ 和 $L_2$ 个类别的分类变量，交互项增加 $(L_1-1)(L_2-1)$ 个示性变量的乘积。

```r
# four equivalent ways to fit Y on X1, X2 and X1*X2
model1 <- lm(Y ~ X1 * X2, data = data)            # "*": main effects + interaction
model2 <- lm(Y ~ X1 + X2 + X1:X2, data = data)    # ":": interaction only
model3 <- lm(Y ~ X1 + X2 + I(X1 * X2), data = data)
model4 <- lm(Y ~ (X1 + X2)^2, data = data)        # all terms up to order 2
```

只有当两个变量都是数值型时，`model3` 才与其他写法等价；对因子应使用 `*` 或 `:`。

## 结果、残差与预测

```r
summary(model)        # coefficients, t-tests, R^2, overall F-test
anova(model)          # sequential (type I) ANOVA table
confint(model, level = 0.95)

resid <- residuals(model)        # e = y - y_hat
fit.val <- fitted(model)         # y_hat
est.val <- predict(model)        # same as fitted(model)
rstandard(model)                 # studentized residuals r_i
hatvalues(model)                 # leverages h_ii
cooks.distance(model)            # Cook's distances D_i
```

`anova(model)` 报告 {prf:ref}`ex-sequential-ss` 中的序贯平方和，它们依赖于预测变量在公式中的顺序。

`predict()` 的 `interval` 参数用于在平均响应的置信区间 {eq}`eq-ci-mean` 与新观测的预测区间 {eq}`eq-pi` 之间选择：

```r
pred.ci <- predict(model, newdata = data_test, interval = "confidence", level = 0.95)
pred.pi <- predict(model, newdata = data_test, interval = "prediction", level = 0.95)
```

## 多重共线性

### 别名：完全共线性

**别名**（aliased）预测变量是其他预测变量的精确线性组合，即完全共线。例如，若 $x_2 = 3x_1 + 1$，则 $x_2$ 是别名变量。`lm()` 把它们的系数报告为 `NA`，在计算 VIF 之前必须将其删除。`alias()` 会列出它们：

```r
model <- lm(Y ~ ., data = data)
alias(model)
```

`alias()` 的部分输出如下：

```text
Complete :
         (Intercept) V2 V3 V4 ... V13 ... frwy frwynos artnos
art       2          -1  0  0 ...  0  ...  1    0       0
age      86           0  0  0 ... -1  ...  0    0       0
ruralnos  0           0  0  1 ...  0  ...  0   -1      -1
```

每一行把一个别名变量表示为其他变量的组合：

$$
\begin{gathered}
\mathrm{art} = 2 - \mathrm{V2} + \mathrm{frwy}, \\
\mathrm{age} = 86 - \mathrm{V13}, \\
\mathrm{ruralnos} = \mathrm{V4} - \mathrm{frwynos} - \mathrm{artnos}.
\end{gathered}
$$

### 方差膨胀因子

```r
# install.packages("car")
library(car)
vif(model)
```

对含因子的模型，`vif()` 对每一项报告广义 VIF。

## 变量选择

### 用 `regsubsets()` 进行子集选择

先安装一次程序包：

```r
install.packages("leaps")
```

`regsubsets()` 的主要参数：

| 参数 | 类型 | 含义 |
|---|---|---|
| `nvmax` | 整数 | 考察的子集的最大规模 |
| `intercept` | 逻辑值 | 是否包含截距 |
| `method` | 字符 | `"exhaustive"`、`"backward"`、`"forward"` 或 `"seqrep"` |
| `force.in` | 索引或逻辑向量 | 每个模型中必须保留的预测变量 |
| `nbest` | 整数 | 每种子集规模下记录的最优模型个数 |

示例使用 `ISLR` 包中的 `Hitters` 数据，它记录了棒球运动员的薪水和比赛统计数据。

```r
library(ISLR)
library(leaps)

names(Hitters)                  # variable names
Hitters <- na.omit(Hitters)     # remove observations with missing values
dim(Hitters)
sum(is.na(Hitters))             # should be 0
```

**最优子集选择。** 对每种规模（最多 19 个预测变量）拟合最优模型：

```r
regfit.full <- regsubsets(Salary ~ ., data = Hitters, nvmax = 19)
reg.summary <- summary(regfit.full)
reg.summary
```

摘要中包含每种规模下最优模型的各项准则，包括 $R^2$（`"rsq"`）、$\mathit{SS}_E$（`"rss"`）、调整 $R^2$（`"adjr2"`）、$C_p$（`"cp"`）和 BIC（`"bic"`）：

```r
names(reg.summary)
#> [1] "which"  "rsq"    "rss"    "adjr2"  "cp"     "bic"    "outmat" "obj"
reg.summary$rsq
```

**绘图。** `regsubsets` 对象的 `plot()` 方法按所选准则排序，显示每种规模下最优模型包含哪些预测变量：

```r
plot(regfit.full, scale = "r2")
plot(regfit.full, scale = "adjr2")
plot(regfit.full, scale = "Cp")
plot(regfit.full, scale = "bic")
```

**系数与协方差矩阵：**

```r
coef(regfit.full, 1:3)   # coefficients of the best models with 1 to 3 predictors
vcov(regfit.full, 1)     # covariance matrix of the best one-predictor model
```

**向前、向后和序贯替换选择：**

```r
regfit.fwd <- regsubsets(Salary ~ ., data = Hitters, nvmax = 19, method = "forward")
summary(regfit.fwd)

regfit.bwd <- regsubsets(Salary ~ ., data = Hitters, nvmax = 19, method = "backward")
summary(regfit.bwd)

regfit.seq <- regsubsets(Salary ~ ., data = Hitters, nvmax = 19, method = "seqrep")
summary(regfit.seq)
```

### 用 `step()` 进行逐步选择

`step()` 函数（以及 `MASS` 包中的 `stepAIC()`）按 AIC 进行逐步选择：每一步做出使 AIC 下降最多的改变，当 AIC 不再下降时停止。对线性模型，它使用 $n\ln(\mathit{SS}_E/n) + 2p$，即 {eq}`eq-aic-linear` 的形式；设置 `k = log(n)` 则改用 BIC。主要参数：

| 参数 | 类型 | 含义 |
|---|---|---|
| `direction` | 字符 | `"both"`、`"backward"` 或 `"forward"` |
| `scope` | 公式或列表 | 搜索的模型范围 |
| `trace` | 逻辑值或整数 | 是否（以及多详细地）输出选择过程 |

```r
lm.inter_only <- lm(Salary ~ 1, data = Hitters)   # intercept-only model
lm.all        <- lm(Salary ~ ., data = Hitters)   # full model

# Forward stepwise selection
step.fwd <- step(lm.inter_only, direction = "forward",
                 scope = formula(lm.all), trace = 1)
summary(step.fwd)

# Backward stepwise selection
step.bwd <- step(lm.all, direction = "backward",
                 scope = formula(lm.all), trace = 1)
summary(step.bwd)
```

## 本章回顾
### 小结

- `lm(Y ~ ..., data = data)` 拟合线性模型；公式控制预测变量、截距、变换（`I()`）、因子和交互项（`*`、`:`）。
- `residuals()`、`fitted()`、`rstandard()`、`hatvalues()` 和 `cooks.distance()` 给出第二部分的诊断量；`predict(..., interval = )` 给出置信区间和预测区间。
- `alias()` 发现完全共线性；`car::vif()` 计算 VIF。
- `leaps::regsubsets()` 实现最优子集、向前、向后和序贯替换选择；`step()` 实现基于 AIC 的逐步选择。

### 常见错误

- 在公式中写 `X1^2` 而不是 `I(X1^2)`；没有 `I()` 时 `^` 表示"最高到二阶的交互"，`X1^2` 就只是 `X1`。
- 把分类变量的数值编码当作连续预测变量；应使用 `factor()`。
- 需要预测单个未来值时却使用 `interval = "confidence"`。
- 对含有别名（`NA`）系数的模型调用 `vif()`。

## 参考文献

- Fox, J., & Weisberg, S. (2019). *An R companion to applied regression* (3rd ed.). Sage.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An introduction to statistical learning: With applications in R*. Springer. <https://doi.org/10.1007/978-1-4614-7138-7>
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *ISLR: Data for an introduction to statistical learning with applications in R* (R package version 1.4) [Computer software]. <https://CRAN.R-project.org/package=ISLR>
- Lumley, T. (2024). *leaps: Regression subset selection* (R package version 3.2) [Computer software]. <https://CRAN.R-project.org/package=leaps>
- R Core Team. (2025). *R: A language and environment for statistical computing* [Computer software]. R Foundation for Statistical Computing. <https://www.R-project.org/>
- Statology. (n.d.). *A complete guide to stepwise regression in R*. <https://www.statology.org/stepwise-regression-r/>

# 术语表

本书使用的中英文术语对照。中文版在每章首次出现某术语时，会在括号中给出其英文名称。

## 预备知识

| 中文 | English | 符号 / 说明 |
|---|---|---|
| 随机变量 / 随机向量 | random variable / random vector | $X$, $\bz$ |
| 累积分布函数 | cumulative distribution function (CDF) | $F$ |
| 概率质量函数 / 概率密度函数 | probability mass / density function | $f$ |
| 期望 / 方差 / 协方差 | expectation / variance / covariance | $\E$, $\Var$, $\Cov$ |
| 条件期望 | conditional expectation | $\E[Y\mid X]$ |
| 协方差矩阵 | covariance matrix | $\Cov(\bz)$ |
| 多元正态分布 | multivariate normal distribution | $\mathcal{N}_m(\boldsymbol\mu,\bSigma)$ |
| 卡方分布 / $t$ 分布 / $F$ 分布 | chi-square / $t$ / $F$ distribution | |
| 非中心分布 / 非中心参数 | noncentral distribution / noncentrality parameter | $\delta$ |
| 二次型 | quadratic form | $\bz\T\mathbf{A}\bz$ |
| Cochran 定理 | Cochran's theorem | |
| Pearson 相关系数 | Pearson correlation coefficient | $r$, $\rho$ |
| Fisher $z$ 变换 | Fisher $z$-transformation | $\operatorname{artanh}(r)$ |
| Spearman / Kendall 秩相关系数 | Spearman / Kendall rank correlation | $R_S$, $\tau$ |
| 一致对 / 不一致对 | concordant / discordant pair | $n_c$, $n_d$ |
| 拟合优度检验 | goodness-of-fit test | |
| 经验分布函数 | empirical distribution function | $F_n$ |
| Kolmogorov–Smirnov 检验（K–S 检验） | Kolmogorov–Smirnov test | $D_n$ |
| 分位数–分位数图（Q–Q 图） | quantile–quantile (Q–Q) plot | |
| 偏差 / 均方误差 | bias / mean squared error (MSE) | $\operatorname{MSE} = \Var + \operatorname{Bias}^2$ |
| 第一类 / 第二类错误，功效 | type I / type II error, power | $\alpha$, $\beta$, $1-\beta$ |
| 大数定律 / 中心极限定理 | law of large numbers / central limit theorem | |
| 秩 / 迹 / 行列式 | rank / trace / determinant | $\rank$, $\tr$, $\det$ |
| 特征值 / 谱分解 | eigenvalue / spectral decomposition | |
| 正定（半正定） | positive (semi)definite | |
| 正交投影 | orthogonal projection | |
| 奇异值分解 | singular value decomposition (SVD) | $\bX = \mathbf{U}\mathbf{D}\mathbf{V}\T$ |
| 梯度 / Hessian 矩阵（海森矩阵） | gradient / Hessian matrix | $\nabla f$ |
| 凸函数 / 次梯度 | convex function / subgradient | $\partial f$ |

## 线性回归

| 中文 | English | 符号 / 说明 |
|---|---|---|
| 响应变量（因变量） | response (dependent variable) | $y$ |
| 预测变量（自变量、解释变量、回归变量、协变量） | predictor (independent / explanatory variable, regressor, covariate) | $x_j$ |
| 外生变量 / 内生变量 | exogenous / endogenous variable | |
| 回归系数 | regression coefficient | $\beta_j$ |
| 截距 | intercept | $\beta_0$ |
| 设计矩阵 | design matrix | $\bX$ |
| 误差 | error | $\varepsilon_i$ |
| 残差 | residual | $e_i$ |
| 拟合值 | fitted value | $\widehat{y}_i$ |
| 参数 / 估计量 / 估计值 | parameter / estimator / estimate | $\bbeta$ / $\widehat{\bbeta}$ / 数值 |
| 同方差性 | homoskedasticity | $\Var(\varepsilon_i)=\sigma^2$ |
| 普通最小二乘（OLS） | ordinary least squares (OLS) | |
| 正规方程 | normal equations | $\bX\T\bX\bbeta=\bX\T\by$ |
| 极大似然估计 | maximum likelihood estimation (MLE) | |
| 无偏估计量 | unbiased estimator | |
| 帽子矩阵 / 投影矩阵 | hat matrix / projection matrix | $\bH$ |
| 幂等 | idempotent | $\bH^2=\bH$ |
| 自由度 | degrees of freedom | |
| 总平方和 / 回归平方和 / 残差平方和 | total / regression / residual sum of squares | $\mathit{SS}_T$, $\mathit{SS}_R$, $\mathit{SS}_E$ |
| 均方 | mean square | $\mathit{MS}$ |
| 方差分析 | analysis of variance (ANOVA) | |
| 标准误 | standard error | $\mathrm{SE}$ |
| 置信区间 | confidence interval | |
| 预测区间 | prediction interval | |
| 额外平方和 | extra sum of squares | $\mathit{SS}_R(\bbeta_1\mid\bbeta_2)$ |
| 决定系数 | coefficient of determination | $R^2$ |
| 调整 $R^2$ | adjusted $R^2$ | $R^2_{\mathrm{adj}}$ |
| 杠杆值 | leverage | $h_{ii}$ |
| 标准化残差 / 学生化残差 | standardized / studentized residual | $d_i$ / $r_i$ |
| 强影响点 | influential observation | |
| Cook 距离 | Cook's distance | $D_i$ |
| 单因素方差分析 | one-way analysis of variance | |
| 处理效应 | treatment effect | $\tau_i$ |
| 固定效应模型 | fixed-effects model | |
| 处理 / 重复 | treatment / replicate | |
| 多重比较 | multiple comparisons | |
| 最小显著差异 | least significant difference (LSD) | $\mathrm{LSD}$ |
| 随机效应模型 / 方差分量 | random-effects model / variance component | $\sigma_\tau^2$ |
| 干扰因素 / 区组 | nuisance factor / block | |
| 随机化完全区组设计 | randomized complete block design (RCBD) | |

## 模型构建

| 中文 | English | 符号 / 说明 |
|---|---|---|
| 标准化回归系数 | standardized regression coefficient | $\beta_j^{*}$ |
| 多重共线性 | multicollinearity | |
| 方差膨胀因子 | variance inflation factor (VIF) | $\mathrm{VIF}_j$ |
| 最优子集选择 | best subset selection | |
| 向前 / 向后逐步选择 | forward / backward stepwise selection | |
| Mallows $C_p$ 统计量 | Mallows' $C_p$ | |
| 赤池信息准则 / 贝叶斯信息准则 | Akaike / Bayesian information criterion | AIC / BIC |
| 上包络线 | upper envelope | |
| 过拟合 | overfitting | |
| 惩罚（正则化）回归 | penalized (regularized) regression | |
| 岭回归 | ridge regression | $\lambda\lVert\bbeta\rVert_2^2$ |
| lasso（套索回归） | lasso | $\lambda\lVert\bbeta\rVert_1$ |
| 弹性网 | elastic net | |
| 调节（惩罚）参数 | tuning (penalty) parameter | $\lambda$ |
| 收缩 | shrinkage | |
| 软阈值 / 硬阈值 | soft / hard thresholding | $S(z,\gamma)$ |
| 坐标下降 | coordinate descent | |
| 有效自由度 | effective degrees of freedom | $\operatorname{df}(\lambda)$ |
| 交叉验证 | cross-validation | |
| 正则化路径 | regularization path | |

## 广义线性模型

| 中文 | English | 符号 / 说明 |
|---|---|---|
| 广义线性模型 | generalized linear model (GLM) | |
| 指数族 | exponential family | |
| 典则参数（自然参数） | canonical parameter | $\theta$ |
| 离散参数 | dispersion parameter | $\phi$ |
| 线性预测子 | linear predictor | $\eta$ |
| 联系函数 / 典则联系函数 | link function / canonical link | $g$ |
| 方差函数 | variance function | $V(\mu)$ |
| 偏差 / 饱和模型 | deviance / saturated model | $D$ |
| Fisher 信息 / Fisher 得分法 | Fisher information / Fisher scoring | |
| 迭代重加权最小二乘 | iteratively reweighted least squares (IRLS) | |
| logistic 回归（逻辑回归） | logistic regression | |
| 优势 / 优势比 | odds / odds ratio | $e^{\beta_j}$ |
| 边际效应 | marginal effect | |
| （完全）分离 | separation | |
| 伪 $R^2$ | pseudo-$R^2$ | |
| 泊松回归 | Poisson regression | |
| 率比（相对风险） | rate ratio (relative risk) | $e^{\beta_j}$ |
| 偏移项 / 暴露量 | offset / exposure | $\ln E_i$ / $E_i$ |
| 偏差下降检验 | drop-in-deviance test | |
| Wald 检验 | Wald test | |
| Pearson 残差 / 偏差残差 | Pearson / deviance residual | |
| 过度离散 | overdispersion | |
| 拟泊松 | quasi-Poisson | |
| 负二项回归 | negative binomial regression | |
| 零膨胀泊松 | zero-inflated Poisson (ZIP) | |
| 有序回归 / 累积联系模型 | ordinal regression / cumulative link model | |
| 比例优势模型 | proportional odds model | |
| 切点（阈值） | cut-point (threshold) | $\theta_j$ |
| 多项 logit / 多项 probit | multinomial logit (MNL) / probit (MNP) | |
| 随机效用最大化 | random utility maximization | |
| 选择集 | choice set | $C_n$ |
| 无关选项独立性 | independence of irrelevant alternatives (IIA) | |
| 似然比指数 | likelihood ratio index | $\rho^2$ |

## 分位数回归

| 中文 | English | 符号 / 说明 |
|---|---|---|
| 分位数 | quantile | $x_\tau$ |
| 分位数回归 | quantile regression | |
| 弹球损失（分位数损失、检验函数） | pinball loss (quantile loss, check function) | $\rho_\tau$ |
| 中位数回归（最小一乘） | median regression (least absolute deviations) | $\tau=0.5$ |

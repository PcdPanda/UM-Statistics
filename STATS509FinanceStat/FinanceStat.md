[toc]

# 1. Preliminaries

### 1.1 收益率

##### 收益率

在收益率很小的情况下,$R_t$和$r_t$非常接近,且$r_t$始终小于$R_t$

- 净收益率 $R_t=\frac{P_t}{P_{t-k}} - 1=e^{r_t}-1$
- 对数收益率 $r_t=\ln\frac{P_t}{P_{t-k}}=\ln(1+R_t)=\ln P_t-\ln P_{t-k}$

##### 分红收益率

本质上就是在当日价格之上添加分红金额

- 分红收益率 $R_t=\frac{P_t+D_t}{P_{t-1}}$
- 分红对数收益率$r_t=\ln\frac{P_t+D_t}{P_{t-1}}$

##### <u>随机游走</u>

- 使用对数收益率,且每个节点的收益率符合i.i.d

- 用中心极限定理,可以得<u>到区间对数收益率分布</u>
  $$
  r_t(k)=\sum_{i=1}^k r_t\sim N(\sum\mu_t, (\sum\sigma_t)^2)
  $$

  - $\mu_t$: $t$时刻的漂移 

- 几何随机游走: 带入区间收益率可以得到价格的分布
  $$
  P_t=P_0\exp(r_t(k))
  $$

### 1.2 Tail Probabilities

##### Tail Probability

给定$x$,随机变量大于$x$的概率(描述了风险)

- $P[X\geq x]=1-F(x)=\int_x^\infty f(x)\mathrm{d}x$
- 通常和指数分布比较,是否是heavy tail

##### Heavy Tail

Tail定义了特殊分布在<u>极端值时出现的概率</u>,如果概率比<u>指数分布</u>大,则为heavy tail
$$
\int_{-\infty}^{\infty}e^{\beta x}\mathrm dF(x)=\infty\quad \text{for all}\quad \beta>0\rightarrow\lim_{x\rightarrow\infty}\frac{\text{Pr}[X>x]}{\text{Pr}[X_{\text{exp}>x}]}=\infty\quad\text{for all}\quad \beta>0
$$

- 通过Q-Q Plot来比较真实数据分布的tail和理想分布
- $f(x)\rightarrow0$的速度,决定了tail的厚度, 多项式tail>指数tail>正态分布tail

### 1.3 <u>Value at Risk (VaR)</u>

##### <u>风险资产 (VaR)</u>

<u>描述了暴露在给定风险下亏损的资产量下限(只有$q$概率亏损超过 VaR$_t$)</u>

| $t$  | $P_t$        | $q$      | $F$        |
| ---- | ------------ | -------- | ---------- |
| 时间 | $t$ 时刻资产 | 亏损概率 | 收益率分布 |

- <u>Absolute VaR</u>
  $$
  \text{P}[\text{VaR}_t<P_t-P_{t+\bigtriangleup t}]=q
  $$

- <u>Relative VaR</u>
  $$
  \text{P}[R_t<-\tilde{\text{VaR}}_t]=q
  $$

  - 本质上是<u>收益率分位数的相反数,亏损率的逆分位数</u>
  - $\tilde{\text{VaR}}=\frac{\text{VaR}}{P_t}$

### <u>1.4 Expected Shortfall (ES)</u>

##### Shortfall Distribution

<u>给定资产分布,描述了亏损超过$\text{VaR}_q$时的分布</u>
$$
\Theta_{X_q}(x)=\text{P}[-X\leq x|X\leq-\text{VaR}_q]=\frac{q-F(-x)}{q}
$$
当收益率分布对称时,$F(-x)=1-F(x)$,则有$F_{X_q}=\frac{F(x)-(1-q)}{q}$

##### <u>Expected Shortfall</u>

<u>描述了当亏损超过$\text{VaR}_q$时,期望的亏损值</u>
$$
ES_q=E[-X|X\leq-\text{VaR}_q]=-\frac{1}{q}\int_{x<-\text{VaR}_q}xf(x)\mathrm d x=
$$

# 2. Distribution Inference

### 2.1 Density Estimation

##### Kernel-Based Estimation

- 给定若干个离散样本$x_i$,构造平滑分布
  $$
  \hat f_b(x)=\frac{1}{n}\sum_{i=1}^nK_b(x-x_i)
  $$

- $K_b$: kernel function,<u>可以是任何density function,方差必须是1</u>

- 本质上是样本的平滑分布,$b$越大,越平滑

##### Bandwidth $b$

- 描述了平滑分布对于离散样本信息的利用程度
- $b$越大,拟合离$K_b$分布越接近,越平滑
- $b$越小,拟合离样本分布越接近,但容易过拟合

##### Kernel-Based Estimation Property

- 期望: $E[\hat f_b(x)]=\frac{1}{n}\sum_{i=1}^n E[K_b(x-x_i)]$
  - 如果$x_i$是iid则有$E[\hat f_b(x)]=(K_b*f)(x)$

- 方差: $Var[\hat f_b(x)]=\frac{1}{n}\sum_{i=1}^n Var[K_b(x-x_i)]$

### 2.2 Test of Normality

##### <u>Shapiro-Wilk test</u>

- 关注分布的Tail,适合金融数据使用

- 将样本从小到大排序,构造$x_1,x_2,\cdots x_n$

- 基于$x$构造协方差矩阵$V$

- 基于$x$构造期望向量$m=\begin{bmatrix}m_1&m_2\cdots m_n\end{bmatrix}$
  $$
  W=\frac{(m^TV^{-1}x)^2}{\sum(x_i-\bar x)^2(m^TV^{-1}V^{-1}m)}
  $$

- 通过蒙特卡洛模拟来检验

##### Jarque-Bera

- 通过样本的偏度$S$和峰度$K$来构造参数检验
  $$
  JB=\frac{n}{6}(S^2+\frac{(K-3)^2}{4})\sim\chi^2_2
  $$

- $JB$值越偏离$\chi^2_2$,说明样本正态分布的概率越小

### 2.3 Pareto Distribution Property

##### General Pareto Distribution

| 参数                | 随机变量$X$         | PDF                                                          | CDF                                                   | 期望                            | 方差                                          |
| ------------------- | ------------------- | ------------------------------------------------------------ | ----------------------------------------------------- | ------------------------------- | --------------------------------------------- |
| $\alpha,x_m,\sigma$ | 和最低值$x_m$的比值 | $f_X(x)=\frac{1}{\sigma}(1+\frac{x-x_m}{\alpha\sigma})^{-\alpha - 1}$ | $F_X(x)=1-(1+\frac{(x-x_m)}{\alpha\sigma})^{-\alpha}$ | $x_m+\frac{\sigma}{1-1/\alpha}$ | $\frac{\sigma^2}{(1-1/\alpha)^2(1-2/\alpha)}$ |

帕累托分布具有<u>特别厚的多项式尾部</u>,适合用来估计条件概率

##### <u>Pickands-Balkema-de-Hann Theorem</u>

- 对于绝大部分分布$f(x)$的条件概率分布,都可以转化为帕里托分布,具有polynominal tail
  $$
  1-F(x)=P[X>u+x|X>u]=\frac{P[X>u+xa(u)]}{P[X>u]}\rightarrow (1+\frac{x}{\alpha})^{-\alpha}\quad(u\rightarrow\infty)
  $$

- 很难计算$a(u)$的值,但是可以用QQ plot判断是否符合Pareto Distribution并分析tail

### 2.4 <u>Tail Estimation</u> 

##### Overview

- KDE不适合估计Tail的分布,因为缺少数据

- 使用条件概率来估计Tail分布
  $$
  P[X\geq x]=P[X\geq x|X\geq x_m]P[X\geq x_m]
  $$

- 使用KDE估计$P[X\geq x_m]$的值,使用Pareto估计$P[X\geq x|X\geq x_m]$

##### 估计流程

1. Plot ECDF分布的Tail (只能使用daily return而不是log return)
2. 使用shape plot画出GPD的$\hat\xi$和threshold $u$的关系
3. threshold要尽量大,但是shape parameter要尽量稳定
4. 通过Tail plot检验threshold选取的质量
5. 计算$\tilde{VaR_q}=GPD^{-1}[1-\frac{q}{1-ECDF(x_m)}]$
6. 基于GPD期望的性质,可以得到$ES=\tilde{VaR_q}+\frac{\hat\sigma+\hat\xi(\tilde VaR_q-x_m)}{1-\hat\xi}$

##### 多变量Tail Dependence

- Lower Dependency (描述左下角): $\lambda_l=\lim_{q\rightarrow0}P[F_1(X_1)\leq q|F_2(X_2)\leq q]$
- Upper Dependency (描述右上角): $\lambda_u=\lim_{q\rightarrow 1}P[F_1(X_1)\geq q|F_2(X_q)\geq q]$
- Gaussian Copula: $\lambda_u=\lambda_l=0$
- T-Copula: $\lambda_u=\lambda_l=2F_{t,\nu+1}[-\sqrt{\frac{(\nu + 1)(1-\rho)}{1+\rho}}]$

# 3. Portfoilo Management

### 3.1 Portfoilo基本性质

##### 双资产管理

给定双资产组成的Portfolio $V=n_1P_1+n_2P_2$

- 权重向量$w=\begin{bmatrix}\frac{n_1P_1}{n_1P_1'+n_2P_2'}&\frac{n_2P_2}{n_1P_1'+n_2P_2'}\end{bmatrix}$

- 收益率$R=\frac{V'}{V}=\frac{n_1P_1'+n_2P_2'}{n_1P_1+n_2P_2}$,通过$R$来计算$VaR$和Expected Shortfall

- 对于双资产Portfolio $V=wR_1+(1-w)R_2$,则<u>通过分散投资不相关资产来最小化风险</u>
  $$
  w=\frac{\sigma^2_2-\rho_{12}\sigma_1\sigma_2}{\sigma_1^2+\sigma_2^2-2\rho_{12}\sigma_1\sigma_2}
  $$

##### <u>Copula模型应用</u>

1. 对每个asset估计marginal distribution $\hat F_P$
2. 使用QQ plot判断$\hat F_p$的估计准确度,注意tail
3. 计算asset之间的spearman相关系数
4. 用$\hat F_p$生成概率并带入copula模型

5. 绘制copula的等高线,分析估计质量
6. 从copula生成模拟数据,并获得不同权重时的$VaR$

	### 3.2 风险资产模型

##### 定义

| 资产收益率向量                                  | 资产收益率期望                                         | 收益率协方差矩阵       |
| ----------------------------------------------- | ------------------------------------------------------ | ---------------------- |
| $R=\begin{bmatrix}R_1\cdots R_N\end{bmatrix}^T$ | $E[R]=\begin{bmatrix}\mu_1\cdots \mu_N\end{bmatrix}^T$ | $\text{Cov}(R)=\Sigma$ |

已知上述条件,找到最合适的权重向量$W=\begin{bmatrix}w_1\cdots w_N\end{bmatrix}^T$在收益率最大的情况下最小化风险

- 最小化标准差: $\min w^T\Sigma w$
- 最大化收益率: $w^T\mu=\mu^*$
- 权重和为1: $w^TI=1$
- 可能还有分散投资,不能做空等附加条件
- 生成的结果为<u>有效边界: 在固定资产的方差$\sigma$的情况下,可以获得的最大收益$\mu$</u>

##### 无风险资产混合投资

- 在资产组合中引入了收益率为$\mu_f$的无风险资产
- 需要最大化夏普比率:描述了额外收益$E(R)-\mu_f$和风险$\sigma$的比值,<u>最大化夏普比率等于寻找切线</u>

$$
\frac{E(R)-\mu_f}{\sigma}
$$

- 令风险资产权重为$w$则有
  - 收益率$\mu=\mu_f+w(\mu_p-\mu_f)$
  - 方差$\sigma=w\sigma_p$

##### 最大化夏普比率

- 对于双资产混合无风险模型,令$v_1=\mu_1-\mu_f,v_2=\mu_2-\mu_f$,则第一个资产的权重有
  $$
  w_T=\frac{v_1\sigma_2^2-v_2\rho_{12}\sigma_1\sigma_2}{v_1\sigma_2^2+v_2\sigma_1^2-(v_1+v_2)\rho_{12}\sigma_1\sigma_2}
  $$

- 在最大化夏普比率的基础上,根据资产模型的方差限制,获得无风险资产的配置

1. 根据$w_T$获得风险资产标准差$\sigma_T,\mu_T$
2. 根据给定的风险目标,获得风险资产总占比权重 $w=\frac{\sigma_{target}}{\sigma_T}$
3. 或者根据给定的收益率目标,获得风险资产总比权重$w=\frac{\mu_{target}-\mu_f}{\mu_T-\mu_f}$
4. 每个风险资产的比重为$w\cdot w_T$






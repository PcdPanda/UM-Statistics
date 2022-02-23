### 基本概念

##### 收益率

今天减$k$天前的价格

- 收益率: $R_t=\frac{P_t}{P_{t-k}} - 1=e^{r_t}-1$

- 对数收益率: $r_t=\ln\frac{P_t}{P_{t-k}}=\ln(1+R_t)=\ln P_t-\ln P_{t-k}$

- 分红收益率: $R_t=\frac{P_t+D_t}{P_{t-1}}$

- 对数分红收益率: $r_t=\ln\frac{P_t+D_t}{P_{t-1}}$

##### VaR

- 相对风险资产(收益率的负分位数): $P[R_t+\tilde{VaR}]=q$
- 风险资产(相对值乘以portfolio大小): 亏损超$VaR$的概率小于$q$, $VaR=P_t\cdot \tilde{VaR_t}$

##### Shortfall Distribution

- 给定亏损分布$F$,则控制亏损在$x$和$VaR_q$中的概率: $\Theta_q(x)=P[X\le x|X>VaR_q]=\frac{F(x)-(1-q)}{q}$
- 亏损大于$VaR$情况下的期望值: $ES_q=E[X|X>VaR_q]=\frac{1}{q}\int_{x>VaR_q}xf(x)\mathrm d x$

收益率不同分布下的ES

| 正态分布 | Dexp                                         | GPD                              |
| -------- | :------------------------------------------- | -------------------------------- |
| $        | $\mu+\sigma\frac{f(F^{-1}(\alpha))}{\alpha}$ | $VaR_q+\frac{\sigma}{\xi q^\xi}$ |

##### 时间序列数据

- 强平稳: 不同时间分布相同
- 弱平稳: 不同时间方差和均值相同
- 自方差: $COV(y_h)=\frac{1}{n}\sum_{t=1}^{n-h}(y_t-\mu)(y_{t+h}-\mu)$

- 自相关性: $\gamma_h=\frac{COV(y_h)}{\sigma^2}$

##### 概率分布性质

- $k$阶中心矩: $\mu_k=E[(X-E[x])^k]$
- 偏度: $\frac{\mu_3}{\mu_2^{3/2}}$
- 峰度: $\frac{\mu_4}{\mu_2^2}-3$
- 期望$E(X)=\int xf_X(x)\mathrm d x$
- 方差$\sigma=\int (x-E(X))f_X(x)\mathrm d x$

- Tail probabilities描述当$x\rightarrow\infty$时$1-F(x)$的变化情况,和指数分布比较: $\lim_{x\rightarrow\infty}e^{tx}\mathrm d F(x)=\infty$ 则heavy tail

### 参数估计

##### Kernel Density Estimate

- 给定若干个离散样本$x_i$,构造平滑分布
  $$
  \hat f_b(x)=\frac{1}{n}\sum_{i=1}^nK_b(x-x_i)
  $$

- $K_b$: kernel function,可以是任何density function,方差必须是1

- 本质上是样本的平滑分布,$b$越大,越平滑

##### Semi-parametric Estimation

- 当阈值$u$足够大时,任何tail都可以转化为GPD:
  $$
  P[X>u+x|X>u]\rightarrow H_{\xi}(x)
  $$

- 估计流程
  1. Plot ECDF分布的Tail
  2. 使用shape plot画出GPD的$\hat\xi$和threshold $u$的关系
  3. threshold要尽量大,但是shape parameter要尽量稳定
  4. 通过Tail plot检验threshold选取的质量
  5. 计算$\tilde VaR_q=GPD^{-1}[1-\frac{q}{1-ECDF(u)}]$
  6. 基于GPD期望的性质,可以得到$ES=\tilde VaR_q+\frac{\hat\sigma+\hat\xi(\tilde VaR_q-u)}{1-\hat\xi}$

### 单变量分布

##### 正态分布 N($\mu,\sigma$)

$$
f_X(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(x-\mu)^2}{2\sigma^2})
$$

##### 指数分布 Exp($\lambda$)

$$
f(x)=\begin{cases}\lambda e^{-\lambda x} & x\ge 0\\0\end{cases}\\
F(x)=1-e^{-\lambda x}
$$

- $\mu=\sigma=\frac{1}{\mu}$

##### 双指数分布 Dexp($\mu,\lambda$)

$$
f(x)=\frac{\lambda}{2}e^{-\lambda|x-\mu|}\\
F(x)=\begin{cases}\frac{1}{2}e^{-\lambda|x-\mu|}&x<\mu\\1-\frac{1}{2}e^{-\lambda|x-\mu|}&x\geq\mu\end{cases}
$$

- $\sigma=\frac{\sqrt{2}}{\lambda}$
- 峰度为3

##### t分布

- $\nu$越大,tail越小, 峰度$\frac{\sigma}{\nu-4}$
- $\sigma=\lambda\sqrt\frac{\nu}{\nu-2}$
- $t_\nu(\mu,\lambda^2)$是$Y=\mu+\lambda t_\nu$的表示,标准差不是$\lambda$

##### Pareto Distribution 

$$
f(x)=\frac{\alpha\mu^\alpha}{x^{\alpha+1}}\\
F(x)=1-\frac{\mu}{x}^\alpha &x\geq\mu
$$

##### GPD$(\xi,\sigma)$

- 多项式级的tail

$$
f(x)=\frac{1}{\sigma}(1+\frac{\xi(x-\mu)}{\sigma})^{-\frac{1}{\xi} - 1}\\
F(x)=1-(1+\frac{\xi(x-\mu)}{\sigma})^{-\frac{1}{\xi}}
$$



##### GED($\mu,\lambda^2\nu$)

$$
f(x)=\kappa_\nu\exp(-\frac{1}{2}|\frac{x}{\lambda_\nu}|^\nu)
$$

- $\nu$越小,tail越heavy
- $\sigma=1$
- 转换后 $Y=\mu+\lambda X$, $\sigma=\lambda$

##### Gamma 分布 $\Gamma(\lambda,\alpha)$

描述了$\alpha$个独立指数分布/Gamma分布的和

- Gamma Function
  $$
  \Gamma(\alpha)=\int_0^\infty x^{\alpha-1}e^{-x}\mathrm d x\\
  f_{\alpha,\lambda}(x) = \begin{cases}\frac{\lambda^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\lambda x}&x\ge0\\0\end{cases}
  $$
  对于正整数$\alpha$,$\Gamma(\alpha)=(\alpha-1)!$

- $\mu=\frac{\alpha}{\lambda}$
- $\sigma=\frac{\sqrt \alpha}{\lambda}$

- <u>Gamma分布可加性</u> $Gamma(\gamma,\alpha_1,)+Gamma(\gamma,\alpha_2)=Gamma(\gamma,\alpha_1+\alpha_2)$
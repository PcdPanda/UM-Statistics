[toc]

# 1. Overview

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
\int_{-\infty}^{\infty}e^{\beta x}\mathrm dF(x)=\infty\quad \text{for all}\quad \beta>0\rightarrow\lim_{x\rightarrow\infty}\frac{\text{Pr}[X>x]}{\text{Pr}[X_{\text{exp}>x}]}=\infty\quad\text{far all}\quad \beta>0
$$

- 通过Q-Q Plot来比较真实数据分布的tail和理想分布

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

### <u>1.3 Expected Shortfall (ES)</u>

##### Shortfall Distribution

<u>给定资产分布,描述了亏损超过$\text{VaR}_q$时的分布</u>
$$
F_{X_q}(x)=\text{P}[-X\leq x|X\leq-\text{VaR}_q]=\frac{q-F(-x)}{q}
$$
当收益率分布对称时,$F(-x)=1-F(x)$,则有$F_{X_q}=\frac{F(x)-(1-q)}{q}$

##### <u>Expected Shortfall</u>

<u>描述了当亏损超过$\text{VaR}_q$时,期望的亏损值</u>
$$
ES_q=E[-X|X\leq-\text{VaR}_q]=-\frac{1}{q}\int_{x<-\text{VaR}_q}xf(x)\mathrm d x
$$

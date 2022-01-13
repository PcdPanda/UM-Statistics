# STATS 509 HW1

Author: Chongdan Pan(pandapcd)

### 1. Problem 1

##### (a)

Since $X$ is double exponential with $\mu=0$ and $\sigma=2$, then we get
$$
\frac{\sqrt{2}}{\lambda}=2\\
f_X(x)=\frac{\lambda}{2}e^{-\lambda|x|}
$$
The $0.01$-quantile should be smaller than $\mu$, therefore, we get
$$
F_X(x)=\frac{1}{2}e^{\frac{\sqrt 2}{2}x}=0.01
$$
Then we can get the quantile of $X$ and transform it to $Y$
$$
x_{0.01}=\sqrt 2\ln 0.01\rightarrow y_{0.01}\approx-8.5
$$

##### (b)

Since $Z=\frac{1}{(X-2)^2}$, then we can get the $F_Z(x)=\textrm{P}[Z\leq x]$, therefore
$$
F_z(x)=\mathrm{P}[\frac{1}{(X-2)^2}\leq x]
$$
if $x\leq 0$, then $F_z(x)=0$, otherwise 
$$
F_Z(x)=\mathrm{P[X-2\geq\frac{1}{\sqrt x}]}=1-F_X(2+\frac{1}{\sqrt x})
$$
Therefore, let $F_X(2+\frac{1}{\sqrt{x}})=0$, we get
$$
\frac{1}{\sqrt x}+2=\sqrt 2\ln1.98\rightarrow x=\frac{1}{(2+\sqrt2\ln 1.98)^2}\approx0.11
$$

### 2. Problem 2

##### (a)

Since $r\sim N(0, 0.025)$ and $r=\ln(R+1)$, we have $\frac{\ln(R+1)}{0.025}\sim N(0, 1)$, therefore
$$
R_{total}=1e8\cdot R_{0.002}=1e8\cdot(e^{0.025\Phi(0.002)}-1)\approx-6946910
$$





### 离散型随机变量分布计算

#### 二项分布（binomial distribution)
* 数学形式
  n : 总实验次数

  k : 实验成功的次数
  
  p : 每一次实验成功的概率
  
  则n次实验成功k次的总概率为：

$$
\begin{align*}
   P(X = k) &= \binom{n}{k} \cdot p^k \cdot (1 - p)^{n - k}, 
            &\quad k = 0, 1, 2, \ldots, n
   \end{align*}
$$

* Python 形式

```python
  import scipy.stats as st
  n, k, p = 10, 5, 0.3
  # 公式计算
  from scipy.special import comb # 计算组合数的函数
  comb(N=n, k=k)*p**k*(1-p)**(n-k)
  # 0.10291934519999994
  
  # biom 函数
  st.binom.pmf(k, n=n, p=p)
  # 0.10291934520000007
```

#### 几何分布（geometric distribution）

* 几何分布是若干次伯努利实验，第一次成功的概率
 ```python
  k, p = 5, 0.3
  st.geom.pmf(n, p)
  # 0.012106082099999993
 ```
#### 泊松分布（Poisson distribution）
 ```python
k, lambda_= 10, 5
st.poisson.pmf(k, mu=lambda_)
# 0.018132788707821854
 ```
### 连续型随机变量分布计算
#### 均匀分布（uniform distribution）

$$
f(x) = \begin{cases} 
\frac{1}{b - a} & \text{if } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases}
$$
 ```python
x0, x1, a, b = 11, 14, 10, 25
st.uniform.cdf(x1, a, b-a) - st.uniform.cdf(x0, a, b-a)
# 0.2
 ```

#### 指数分布（exponential distribution）

```python
# 假设热水器100小时内维修的概率遵从 labmda = 0.002 的指数分布
# 求在 300 到 500 小时内需要维修的概率是多少？
x0, x1, lambda_ = 300, 500, 0.002
st.expon.cdf(x1, 0, scale=1/lambda_) - st.expon.cdf(x0, 0, scale=1/lambda_)
# 0.18093219492258417
```

#### 正态分布（normal distribution）

```python
# 计算 p 值
# 设 X ～ N(10, 4)
# 求 P(8<X<14)
mu, sigma = 10, 2
st.norm.cdf(14, mu, sigma) - st.norm.cdf(8, mu, sigma)
# 0.8185946141203637

# 计算 Z 值
st.norm.ppf(0.975, 0, 1)
# 1.959963984540054
```

####  卡方分布（chi-square distribution）

```py
# 计算 p 值
x, df = 39.36407702660391, 24
st.chi2.cdf(x, df)
# 0.9750000000000001

# 计算 Z 值
p, df = 0.975, 24
st.chi2.ppf(p, df)
# 39.36407702660391
```

#### t 分布 （t-distribution）

```py
# 计算 p 值
x, df = 2.3060041350333704, 8
st.t.cdf(x, df)
# 0.9749999999933345

# 计算 Z 值 
p, df = 0.975, 8
st.t.ppf(p, df)
# 2.3060041350333704
```

#### F 分布（F-distribution）
```py
# 计算 p 值   
x, m, n = 2.4470637479798225, 8, 20
st.f.cdf(x, m, n)

# 计算 Z 值
p, m, n = 0.95, 8, 20
st.f.ppf(p, m, n)
```
#### 自定义概率密度分布函数
设 X 的分布密度函数为：
$$
f(x) = \begin{cases}
\frac{1}{6}x & \text{if } 0 \leq x < 3 \\
2 - \frac{x}{2} & \text{if } 3 \leq x \leq 4 \\
0 & \text{otherwise}
\end{cases}
$$
$$
求P\{1 < x \leq \frac{7}{2}\}.
$$

```python3
import scipy.integrate as integrate
# 定义自定义概率密度函数
def my_pdf(x):
    func_1 = lambda x: x/6
    func_2 = lambda x: 2-x/2
    func_3 = lambda x: 0
    return func_1 if 0<=x<3 else func_2 if 3<=x<=4 else func_3

# 定义自定义概率分布函数
def my_cdf(x):
    func_1 = lambda x: x/6
    func_2 = lambda x: 2-x/2
    func_3 = lambda x: 0
    p = 0
    if 0<=x<3:
        p = integrate.quad(func_1, 0, x)[0] 
    elif 3<=x<=4:
        p = integrate.quad(func_2, 3, x)[0] + integrate.quad(func_1, 0, 3)[0]
    else:
        p = 0
    return 1 if p >= 1 else p

# 计算结果
my_cdf(7/2) - my_cdf(1)
# 0.8541666666666666
```

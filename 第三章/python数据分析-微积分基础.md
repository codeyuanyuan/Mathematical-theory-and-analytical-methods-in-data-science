在这个信息化时代，数据分析在各领域中发挥着越来越重要的作用。大家使用大数据技术从海量数据中挖掘信息，发现规律，探索潜在价值。在大数据的研究和应用中，数学是坚实的理论基础。在数据预处理、分析与建模、模型评价与优化等过程中，数学方法扮演着至关重要的角色。

所以接下来，我们用python语言去实现微积分的一些基础计算等。常用第三方SymPy库来实现微积分计算。

SymPy的全称为Symbolic Python，是由纯Python语言编写的**一个用于符号运算的库，能够与其他科学计算库相结合**。符号化的计算采用的是数学对象符号化的计算方式，使用数学对象的精确标识，而不是近似的，**计算结果可以为一个数学表达式**。它的目标在于成为一个富有特色的计算机代数系统，同时保证自身的代码尽可能简单，且易于理解，容易扩展。

### 一、求一元二次方程

使用solve(f, *symbols, **flags)

```python
from sympy import *#导入SymPy库
x  = symbols('x')#将x定义为符号变量
#通过SymPy库的solve()命令求得一元二次方程的两个根
x = solve(x**2-5*x+6,x)
print('一元二次方程的两个根：',x)
```

输出结果：

```python
一元二次方程的两个根：[2, 3]
```

### 二、集合的运算

```python
A = set('12345')#定义集合A
B = set('234')#定义集合B
print('集合A和集合B的并：',A|B)
print('集合A和集合B的交：',A&B)
print('集合A和集合B的差：',A-B)
```

输出结果：

```python
集合A和集合B的并：{'3', '4', '5', '2', '1'}
集合A和集合B的交： {'2', '3', '4'}
集合A和集合B的差：{'1', '5'}
```

### 三、 数列的极限

计算数列的极限，可以使用SymPy库的limit函数实现。语法格式如下：

```python

sympy.limit(e,z,z0,dir='+')
#e：接受SymPy表达式，表示需要进行求极限的数列的一般项或函数。无默认值
#z:接受SymPy表达式，表示需要进行极限的数列的项或函数的目标值。无默认值
#z0:接受any expression,包含所有类型的数值、∞和-∞等，表示自变量趋于有限值或趋于无穷大，其中∞表示无穷大。无默认值
#dir:接受+或-。取值+时，表示趋于有限值的右极限（z→z0+）；取值-时，表示趋于有限值的左极限（z→z0-）。对于无穷大的z0(∞或-∞)，dir参数无效。默认为+
```

#### 3.1、数列1/2,2/3,3/4，…,n/(n+1),...的一般项为$x_n=n/(n+1)$，当$n\to\infty$时，判断数列｛$x_n$｝是否收敛。

```python
from sympy import *#导入SymPy库
n = symbols('n')#将n定义为符号变量
s = n/(n+1)
print('数列的极限为：',limit(s,n,oo))
```

结果输出：

```python
数列的极限为：1
```

#### 3.2、当$x\to -1/2$时，计算函数$f(x)=(1-4x^2)/(2x+1)$的极限。

```python
from sympy import *#导入SymPy库
x = symbols('x')#将x定义为符号变量
s = (1-4*x**2)/(2*x+1)
print('函数的极限：',limit(s,x,-1/2))
```

结果输出：

```python
函数的极限：2.00000000000000
```

#### 3.3、当$x\to \infty$时，计算函数$f(x)=(1+x^3)/2x^3$的极限。

```python
from sympy import *#导入SymPy库
x = symbols('x')#将x定义为符号变量
s = (1+x**3)/(2*x**3)
print('函数的极限：',limit(s,x,oo))
```

结果输出：

```python
函数的极限：1/2
```

### 四、 求导数

求导数可以使用SymPy库的diff函数实现。语法格式如下：

```python
sympy.diff(f,*symbols,**kwargs)
#f:接收SymPy表达式，表示需要进行求导的函数。无默认值
#*symbols:接收symbol，表示需要进行求导的函数的自变量。无默认值
#**kwargs:接收int，表示函数需要求导的阶数。默认为1。
```

#### 4.1 求常数函数$y=C$（其中，C为常数）的导数。

```python
from sympy import *
x = symbols('x')
c=2
y=c
diff(y,x)
```

结果输出：

```python
0
```

#### 4.2、求幂函数$y=x^μ$（其中，μ是常数）的导数。

```python
from sympy import *
x = symbols('x')#将x定义为符号变量
mu= symbols('mu')#将mu定义为符号变量
y=x**mu
init_printing()#美化输出公式
diff(y,x)
```

结果输出：
$$
\mu x^{\mu}/x
$$


#### 4.3、求指数函数$y=a^x$（其中，$a$是常数，且$a>0,a≠1$）的导数。

```python
from sympy import *
x = symbols('x')#将x定义为符号变量
a = symbols('a')#将a定义为符号变量
y=a**x
diff(y,x)
```

结果输出:
$$
a^xlog(a)
$$

#### 4.4、求对数函数$y=log_ax$（其中，$a$是常数，且$a>0,a≠1$）的导数。

```python
from sympy import *
x = symbols('x')#将x定义为符号变量
a = symbols('a')#将a定义为符号变量
y=log(x,a)
init_printing()#美化输出公式
diff(y,x)
```

结果输出：
$$
\frac{1}{xloga} 
$$

#### 4.5、求正弦函数$y=sinx$的导数。

```python
from sympy import *
x = symbols('x')#将x定义为符号变量
y = sin(x)
init_printing()#美化输出公式
diff(y,x)
```

结果输出：
$$
cosx
$$

#### 4.6、求反正弦函数$y=arcsinx$的导数。

```python
from sympy import *
x = symbols('x')#将x定义为符号变量
y = asin(x)
init_printing()#美化输出公式
diff(y,x)
```

结果输出：
$$
\frac{1}{\sqrt{1-x^2} }
$$

### 五、 函数求导法则

```python
from sympy import *
#函数和的导数
x = symbols('x')#将x定义为符号变量
u = log(x,2)
v = x**2+1
y = u+v
init_printing()#美化输出公式
diff(y,x)
```

$$
2x+\frac{1}{xlog2}
$$

```python
#函数差的导数
y1 = u-v
diff(y1,x)
```

$$
-2x+\frac{1}{xlog2}
$$

```python
#函数积的导数
y2 = u*v
diff(y2,x)
```

$$
2xlogx/log2+\frac{x^2+1}{xlog2}
$$

```python
#函数商的导数
y3 = u/v
init_printing()#美化输出公式
diff(y3,x)
```

$$
-\frac{2xlogx}{(x^2+1)^2log2}+\frac{1}{x(x^2+1)log2}
$$

### 六、 复合函数求导法则

#### 6.1、若$y=sinx2$，求$dy/dx$。

```python
#求复合函数的导数
from sympy import *
x = symbols('x')#将x定义为符号变量
u = symbols('u')#将u定义为符号变量
u = x**2
y = sin(u)
init_printing()#美化输出公式
diff(y,x)
```

或者

```python
#求复合函数的导数
from sympy import *
x = symbols('x')#将x定义为符号变量
y = sin(x**2)#不用写出符合函数的分解，直接求导
init_printing()#美化输出公式
diff(y,x)
```

结果输出
$$
2xcosx^2
$$

#### 6.2、若$y=lntanx$，求$dy/dx$。

```python
from sympy import *
x = symbols('x')#将x定义为符号变量
y = log(tan(x))#不用写出符合函数的分解，直接求导
init_printing()#美化输出公式
diff(y,x)
```

结果输出：
$$
(tan^2x+1)/tanx
$$

#### 6.3、若$y=lncos(e^x)$，求$dy/dx$。

```python
from sympy import *
x = symbols('x')#将x定义为符号变量
y = log(cos(exp(x)))#不用写出符合函数的分解，直接求导
init_printing()#美化输出公式
diff(y,x)
```

结果输出：
$$
-\frac{e^xsine^x}{cose^x}
$$

### 七、 求微分

#### 7.1、若$y=sin(2x+1)$，求$dy$。

```python
from sympy import *
x = symbols('x')#将x定义为符号变量
y = sin(2*x+1)
init_printing()#美化输出公式
diff(y,x)
```

结果输出 ：
$$
2cos(2x+1)
$$

#### 7.2、$y=ln(x+\sqrt{x^2+1})$，求$dy$。

```python
from sympy import *
x = symbols('x')#将x定义为符号变量
y =log(x+sqrt(x**2+1))
diff(y,x)
```

结果输出：
$$
\frac{1}{\sqrt{x^2+1}}
$$

### 八、 微分的近似运算

需要numpy库来计算数值。

#### 8.1、求$sin29$的近似值。

```python
import numpy as np
x =(29/360)*2*np.pi#假设x=29°
y=np.sin(x)
print('29°角的正弦数值为',y)
```

结果输出：

```python
29°角的正弦数值为 0.48480962024633706
```

#### 8.2、求$1.02^{1/3}$的近似值。

```python
import numpy as np
x =1.02
y=x**(1/3)
print('1.02开3次方根的值为：',y)
```

结果输出：

```python
1.02开3次方根的值为：1.006622709560113
```

### 九、 微分中值

#### 9.1、求曲线$f(x)=2x3−12x2+18x−2$的凹凸区间及拐点。

```python
from sympy import *
x = symbols('x')#将x定义为符号变量
y =2*x**3-12*x**2+18*x-2
df1=diff(y,x)#一阶导数
df2=diff(y,x,2)#二阶导数
print('令二阶导函数为零的x取值为',solve(df2,x))#solve求解二阶导数方程的解
print('函数在拐点的值为',y.subs(x,2))#把x=2代入y中，算出拐点值
```

结果输出：

```python
令二阶导函数为零的x取值为 [2]
函数在拐点的值为 2
```

#### 9.2、求函数$f(x)=(x+3)^2*(x−1)^3$的极值。

```python
from sympy import *
x = symbols('x')#将x定义为符号变量
y = (x+3)**2*(x-1)**3
df=diff(y,x)#一阶导数
print('函数的驻点:',solve(df,x))#solve求解一阶导数方程的解
print('函数的极值为:',y.subs(x,-3),y.subs(x,-7/5),y.subs(x,1))#把x=-3代入y中，算出极值
```

结果输出：

```python
函数的驻点: [-3, -7/5, 1]
函数的极值为: 0 -35.3894400000000 0
```

#### 9.3、求函数$f(x)=2x^3−6x^2+7$的极值。

利用第二充分条件求函数的极值点

```python
from sympy import *
x = symbols('x')#将x定义为符号变量
y = 2*x**3-6*x**2+7
df = diff(y,x)#一阶求导
solve(df,x)#solve求解一阶导数方程的解
print('函数的极值点:',solve(df,x))
df2 = diff(y,x,2)#二阶求导
print('二阶导数在驻点的值为：',df2.subs(x,0),df2.subs(x,2))#把x=0/2代入y中，算出极值
print('函数的极值为:',y.subs(x,0),y.subs(x,2))
```

结果输出：

```python
函数的极值点: [0, 2]
二阶导数在驻点的值为：-12 12
函数的极值为: 7 -1
```

### 十、最值问题

在工农业生产中，常常会遇到在一定条件下怎么使“产量最多”“用料最少”“成本最低”“效率最高”等问题，这类问题通常称为优化问题，在数学上有时可归结为求某函数（通常称为目标函数）的最大值或最小值问题。

#### 10.1、某公司决定通过增加广告投入和技术改造投入来获得更大的收益。通过对市场的预测，每投入$x$万元广告费，增加的销售额可近似用函数$y_1=−2x^2+14x$（万元）来计算；每投入$x$万元技术改造费，增加的销售额可近似用函数$y_2=-1/3x^3+2x^2+5x$（万元）来计算。该公司准备投入3万元，分别用于广告投入和技术改造投入，如何分配资金才能使该公司获得最大收益？

解：设技术改造投入x万元，即广告投入3-x万元，代入方程：

$f(x)=-2(3-x)^2+14(3-x)-1/3x^3+2x^2+5x$，

简化方程$f(x)=-1/3x^3+3x+24(0≤x≤3)$

```python
from sympy import *
x = symbols('x')#将x定义为符号变量
y = -1/3*x**3+3*x+24
df = diff(y,x)#求一阶导数
solve(df,x)#求解一阶导数方程
print('函数的驻点(或不可导)为：',solve(df,x))
max(y.subs(x,0),y.subs(x,sqrt(3)),y.subs(x,3))#把x等于0，3，根号3代入y中，求出最大值
```

结果输出：

```python
函数的驻点(或不可导)为：[-1.73205080756888, 1.73205080756888]
```

$$
2.0\sqrt{3}+24
$$

### 十一、不定积分

计算连续函数的不定积分，可以使用SymPy库的integrate函数实现。语法格式如下：

```python
sympy.integrate(f,var,…)
# f:接收SymPy表达式，表示需要进行求积分的函数，无默认值
# var:接收symbol、tuple(symbol,a,b)或several variables。symbol表示需要进行求积分的函数的一个自变量；
# tuple(symbol,a,b)用于求定积分，其中symbol表示函数的自变量，a表示积分下限，b表示积分上限；
# several variables表示指定几个变量，在这种情况下，结果是多重积分；
# 若完全不指定var,则返回f的完全反导数，将f整合到所有变量上
```

**利用SymPy库计算不定积分时，最后的输出结果会少一个常数C**，但不影响对问题的理解。

#### 11.1、求不定积分$\int cosxdx$,$\int \frac{1}{1+x^2}dx$,$\int 3x^2dx$。

```python
from sympy import *
x = symbols('x')#将x定义为符号变量
f1 = cos(x)
f2 = 1/(1+x**2)
f3 = 3*x**2
init_printing()#美化输出公式
print("不定积分：",integrate(f1,x),integrate(f2,x),integrate(f3,x))
```

结果输出：

```python
不定积分：sin(x) atan(x) x**3
```

#### 11.2、已知某曲线上的任意一点$P(x,y)$处的切线斜率为该点横坐标的倒数，且该曲线过点$(e^2,3)$，求此曲线方程。

```python
from sympy import *
x = symbols('x')#将x定义为符号变量
f = 1/x
init_printing()#美化输出公式
integrate(f,x)#求不定积分
print(integrate(f,x))
```

结果输出：

```python
log(x)
```

验证不定积分的性质：

#### 11.3.1、验证等式$\int (e^x-3cosx)dx=\int e^xdx+\int(-3cosx)dx$。

```python
from sympy import *
x = symbols('x')#将x定义为符号变量
f = exp(x)
g = -3*cos(x)
init_printing()#美化输出公式
print('左侧结果：',integrate(f+g,x))#等式左侧
init_printing()#美化输出公式
print('右侧结果：',(integrate(f,x)+integrate(g,x)))#等式右侧
init_printing()#美化输出公式
```

结果输出：

```python
左侧结果：exp(x) - 3*sin(x)
右侧结果：exp(x) - 3*sin(x)
```

#### 11.3.2、验证等式$\int 3lnxdx=3\int lnxdx$。

```python
from sympy import *
x = symbols('x')#将x定义为符号变量
f = log(x)
k = 3
init_printing()#美化输出公式
print('左侧结果：',integrate(k*f,x))#等式左侧
init_printing()#美化输出公式
print('右侧结果：',k*(integrate(f,x)))#等式右侧
```

结果输出：

```python
左侧结果：3*x*log(x) - 3*x
右侧结果：3*x*log(x) - 3*x
```

#### 11.4、验证等式$[\int(x-5)^3dx]^{\prime}=(x-5)^3$。

```python
from sympy import *
x = symbols('x')#将x定义为符号变量
f = (x-5)**3
init_printing()#美化输出公式
diff(integrate(f,x),x)
```

结果输出：

```python
x**3 - 15*x**2 + 75*x - 125
```

**并非所有初等函数的不定积分都能求出来，初等函数的原函数未必时初等函数。**

### 十二、定积分

计算定积分的关键是求被积函数的一个原函数，这里只需计算不定积分，所以，也可以使用SymPy库中的integrate函数实现定积分的计算。

#### 12.1、计算由曲线$y=x^2+1$、直线$x=a$和$x=b(a<b)$及$x$轴围成的图形面积。

```python
rom sympy import *
x = symbols('x')#将x定义为符号变量
a = symbols('a')#将a定义为符号变量
b = symbols('b')#将b定义为符号变量
y = x**2+1
print(integrate(y,(x,a,b)))#tuple(symbol,a,b)用于求定积分，其中symbol表示函数的自变量，a表示积分下限，b表示积分上限；
```

结果输出：

```python
-a**3/3 - a + b**3/3 + b
```


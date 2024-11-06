### 一、实验目的及要求

掌握行列式的应用。
掌握矩阵运算的计算机语言描述。

### 二、实验主要内容

1、已知等差序列中前 n 项的和为 $S_n=an^2+bn (n\in N)$, 假设等差序列 $\{a_n\}$的前m项的和为30,前2m项的和为100,求它的前3m项的和。


根据等差数列求和公式性质：在等差数列中，若 $S_m$ 为该数列的前m项和，$S_{2m}$ 为该数列的前2m项和，$S_{3m}$ 为该数列的前3m项和，则 $S_m$,$S_{2m}-S_m$, $S_{3m}-S_{2m}$也为等差数列。

$$
\because \left\{\begin{matrix}
S_m=30
 \\
S_{2m}-S_m=70
 \\
S_{3m}-S_{2m}=110
\end{matrix}\right.
\therefore S_{3m}=210
\\
用行列式检验方程是否有解，是否存在a、b满足： \left\{\begin{matrix}
a+b=30
 \\
4a+2b=1000
 \\
9a+3b=210
\end{matrix}\right.
$$
代码如下：
```python
import numpy as np
D = np.array([[1, 1, 30], [4, 2, 100], [9, 3, 210]])  # 创建三阶行列式
de = np.linalg.det(D)
if de!=0:
    print('行列式为:',de," 行列式不为0所以方程存在解，计算正确")
else:
    print('行列式为:',de," 行列式为0所以方程不存在解，计算错误")
```

![](C:\Users\76129\Desktop\NJUST\2023-2024\ELSE\出书\2-1.png)

2.将坐标轴上的点$P(x,y)$经过矩阵$\begin{pmatrix}
  a&b \\
  c&d
\end{pmatrix}$变换得到新的点$P^{\prime}\left(x^{\prime}, y^{\prime}\right)$,称作一次运
动，即$\begin{pmatrix}
 x^{\prime}\\y^{\prime}
\end{pmatrix}=\begin{pmatrix}
  a&b \\
  c&d
\end{pmatrix}\begin{pmatrix}
 x\\y
\end{pmatrix}$若将点$P(3,4)$ 经过矩阵$A=\begin{pmatrix}
  0&1 \\
  1&0
\end{pmatrix}$变换后得到的新的点$P^{\prime}$,求$P^{\prime}$的坐标。
由题意可知，直接构建矩阵做矩阵乘法运算，代码如下：

```python
import numpy as np
A = np.mat([[0, 1], [1, 0]])
P = np.mat([[3], [4]])
print('矩阵A与P的乘积为:\n', np.dot(A, P))
```
![](C:\Users\76129\Desktop\NJUST\2023-2024\ELSE\出书\2-2.png)
所以坐标为$(4,3)$。

3、某工厂生产甲、乙、丙三种产品，每种产品单位产量的成本与各季度产量如表 1 和表 2 所示。请分别以图的形式直观的展示每个季度中每类成本的总数、每个季度三类成本 总和 4 个季度每类成本的总数。
![](C:\Users\76129\Desktop\NJUST\2023-2024\ELSE\出书\2-3.png)
![](C:\Users\76129\Desktop\NJUST\2023-2024\ELSE\出书\2-4.png)

4、设有甲、乙、丙 3 种酒，主要成分 $\mathrm{A}$、 $\mathrm{B}$、$\mathrm{C}$的各自含量表如表 3 所示。调酒师现 要用这3种酒配出另一种酒，使其中$\mathrm{A}$、 $\mathrm{B}$、$\mathrm{C}$的含量分别为 66.5%、18.5%、15%，请问能 否配出合乎要求的酒？如果能，3 种酒的比例如何分配？当甲酒缺货时，能否用 3 种主要成 分含量为（0.8,0.12,0.08）的丁酒代替。
![](C:\Users\76129\Desktop\NJUST\2023-2024\ELSE\出书\2-5.png)
$$
列出方程组为：\left\{\begin{matrix}
0.7x_1+0.6x_2​+0.65x_3​=0.665 
\\
0.2x_1+0.2x_2​+0.15x_3​=0.185
 \\
0.1x_1+0.2x_2​+0.2x_3​=0.150
\end{matrix}\right.
$$
![](C:\Users\76129\Desktop\NJUST\2023-2024\ELSE\出书\2-6.png)
根据克莱姆法则，得：

![](C:\Users\76129\Desktop\NJUST\2023-2024\ELSE\出书\2-7.png)

![](C:\Users\76129\Desktop\NJUST\2023-2024\ELSE\出书\2-8.png)

代码
```python
import numpy as np
arr = np.mat([[0.7, 0.6, 0.65], [0.2, 0.2, 0.15], [0.1, 0.2, 0.2]])
arr1 = np.mat([[0.665, 0.6, 0.65], [0.185, 0.2, 0.15], [0.15, 0.2, 0.2]])
arr2 = np.mat([[0.7, 0.665, 0.65], [0.2, 0.185, 0.15], [0.1, 0.15, 0.2]])
arr3 = np.mat([[0.7, 0.6, 0.665], [0.2, 0.2, 0.185], [0.1, 0.2, 0.15]])
D = np.linalg.det(arr)
D1 = np.linalg.det(arr1)
D2 = np.linalg.det(arr2)
D3 = np.linalg.det(arr3)
print('甲酒比例', D1/D)
print('乙酒比例', D2/D)
print('丙酒比例', D3/D)

a = np.mat([[0.8, 0.8, 0.8], [0.12, 0.12, 0.12], [0.8, 0.8, 0.8]])
D = np.linalg.det(a)
if D==0:
    print('若用 3 种主要成 分含量为（0.8,0.12,0.08）的丁酒代替 行列式为0 方程无解')
```

5.设有方程组$\left\{\begin{matrix}
x_1-2x_2+2x_3-x_4=1
\\
2x_1-x_2+8x_3=2
 \\
-2x_1+4x_2-2x_3+3x_4=3
\\
3x_1-6x_2-6x_4=4
\end{matrix}\right.$，判断方程组是否有解。若有解，则请求出方程组的全部解。

代码：
```python
import numpy as np
a = np.mat([[1, -2, 2, -1], [2, -1, 8, 0], [-2, 4, -2, 3], [3, -6, 0, -6]])
r_a = np.linalg.matrix_rank(a)
if r_a < 4:
    print("向量组a的秩为：", r_a, "小于4所以4元齐次方程组有非零解")
    b = np.mat([[1, -2, 2, -1, 1], [2, -1, 8, 0, 2],
                [-2, 4, -2, 3, 3], [3, -6, 0, -6, 4]])
    r_b = np.linalg.matrix_rank(b)
    if r_b == r_a and r_a == 4:
        print('向量组b的秩为：4，非齐次线性方程有唯一解')
        B = np.mat([[1], [2], [3], [4]])
        try:
            # print(np.linalg.inv(a))
            x = np.linalg.solve(a, B)
        except:
            print("矩阵a不存在逆矩阵,方程无解")
        else:
            print('唯一解为:', x)
    elif r_a != r_b:
        print('a的秩为：', r_a, 'b的秩为：', r_b, ' 方程无解')
    elif r_b < 4:
        print('方程有无穷个解')
elif r_a == 4:
    print('向量组a的秩为：4,所以4元齐次方程组有零解')
else:
    print('4元齐次方程组无解')
```
![](C:\Users\76129\Desktop\NJUST\2023-2024\ELSE\出书\2-9.png)

6.设某国每年有比例为$p$的农村居民移居城镇，有比例为$q$的城镇居民移居农村。假设该国的总人口数不变，且人口迁移规律不变，将$n$年后农村人口和城镇人口占总人口的比例依次记为$x_{n}$和$y_{n}$, 且 $x_{n}+y_{n}=1$。

(1)求关系式$\begin{pmatrix}
 x_{n+1}\\y_{n+1}
\end{pmatrix}=A\begin{pmatrix}
 x\\y
\end{pmatrix}$中的矩阵$A$。

(2)设目前的农村人口和城镇人口相等，即$\begin{pmatrix}
 x_{0}\\y_{0}
\end{pmatrix}=\begin{pmatrix}
 0.5\\0.5
\end{pmatrix}$，求$\begin{pmatrix}
 x_{n}\\y_{n}
\end{pmatrix}$。

代码
```python
import numpy as np
A = np.mat([[1,2,2],[2,1,2],[2,2,1]])
A1,A2 = np.linalg.eig(A)
print('矩阵A的特征值为：', A1)
print('矩阵A的特征向量为：\n', A2)
# 构造矩阵D
D = np.diag(A1)
print('矩阵D为：\n', D)
# 构建矩阵C
C = np.mat(A2)
print('矩阵C为：\n', C)
```
![](C:\Users\76129\Desktop\NJUST\2023-2024\ELSE\出书\2-10.png)

7.已知矩阵$A=\begin{pmatrix}
  1&2&2 \\
  2&1&2 \\
  2&2&1
\end{pmatrix}$，求$A^k$（其中$k$为正整数）。

代码：
```python
import numpy as np
a = np.mat([[1,2,2],[2,1,2],[2,2,1]])
print('a矩阵为:\n',a)
k = int(input('请输入一个正整数k: '))
print('a^k: \n',np.power(a, k))
```
![](C:\Users\76129\Desktop\NJUST\2023-2024\ELSE\出书\2-11.png)

8.设有一个$3 \times 4$图像的像素矩阵
$A=\begin{pmatrix}
  1&2&1&3 \\
  4&3&2&5 \\
  6&2&1&0
\end{pmatrix}$,请使用奇异值分解法将该图像进行压缩传输。

代码
```python
import numpy as np
A = np.mat([[1, 2, 1, 3], [4, 3, 2, 5], [6, 2, 1, 0]])
U, X, V = np.linalg.svd(A)
print('矩阵A的左奇异向量为：\n', U)
print('矩阵A的奇异值为：', X)
print('矩阵A的右奇异向量为：\n', V)
```
![](C:\Users\76129\Desktop\NJUST\2023-2024\ELSE\出书\2-12.png)

### 三、实验仪器设备
略。

### 四、实验步骤
略。

### 五、遇到的问题与解决办法
![](C:\Users\76129\Desktop\NJUST\2023-2024\ELSE\出书\2-13.png)

刚开始就知道这个异常是矩阵a不可逆，一直以为代码错了，原来是定理记错 了，是比较a和b矩阵的秩，相等才有解，而不是和n，阶数比较：
![](C:\Users\76129\Desktop\NJUST\2023-2024\ELSE\出书\2-14.png)
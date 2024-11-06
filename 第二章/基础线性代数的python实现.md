



### 1.行列式

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
```

行列式的计算 → numpy.linalg.det

计算任何数组a的行列式，这里要求的数组最后两个维度必须是方阵

```python
d = np.array([
    [4, 1, 2, 4],
    [1, 2, 0, 2],
    [10, 5, 2, 0],
    [0, 1, 1, 7]
])
print(d)

#计算行列式
np.linalg.det(d)
```

```python
[[ 4  1  2  4]
 [ 1  2  0  2]
 [10  5  2  0]
 [ 0  1  1  7]]





-1.5099033134902104e-14
```

```python
def createD(a, x, n):
    # 构建函数，生成 n 阶行列式，对角线为x，其余为a
    di = np.eye(n)
    di = di*x
    di[di == 0] = a
    return di
d = createD(5, 20, 10)
print(d)
np.linalg.det(d)
```

```python
[[20.  5.  5.  5.  5.  5.  5.  5.  5.  5.]
 [ 5. 20.  5.  5.  5.  5.  5.  5.  5.  5.]
 [ 5.  5. 20.  5.  5.  5.  5.  5.  5.  5.]
 [ 5.  5.  5. 20.  5.  5.  5.  5.  5.  5.]
 [ 5.  5.  5.  5. 20.  5.  5.  5.  5.  5.]
 [ 5.  5.  5.  5.  5. 20.  5.  5.  5.  5.]
 [ 5.  5.  5.  5.  5.  5. 20.  5.  5.  5.]
 [ 5.  5.  5.  5.  5.  5.  5. 20.  5.  5.]
 [ 5.  5.  5.  5.  5.  5.  5.  5. 20.  5.]
 [ 5.  5.  5.  5.  5.  5.  5.  5.  5. 20.]]





2498818359374.998
```

### 2.矩阵

```python
np.arange(50).reshape(10, 5)
```

```python
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24],
       [25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34],
       [35, 36, 37, 38, 39],
       [40, 41, 42, 43, 44],
       [45, 46, 47, 48, 49]])
```

```python
print(np.eye(10)*10)
np.eye(5)
```

```python
[[10.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0. 10.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0. 10.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0. 10.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0. 10.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0. 10.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0. 10.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0. 10.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0. 10.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0. 10.]]





array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])
```

```python
a = np.arange(10)
print(a)
print(a.shape)
```

```python
[0 1 2 3 4 5 6 7 8 9]
(10,)
```

```python
a = np.arange(10).reshape(1, 10)
b = np.arange(10).reshape(10, 1)
print(a)
print(a.shape)
print(b)
print(b.shape)
```

```python
[[0 1 2 3 4 5 6 7 8 9]]
(1, 10)
[[0]
 [1]
 [2]
 [3]
 [4]
 [5]
 [6]
 [7]
 [8]
 [9]]
(10, 1)
```

#### 2.1矩阵运算

```python
ar1 = np.arange(12).reshape(3, 4)
ar2 = np.arange(10, 22).reshape(3, 4)
ar3 = np.ones((3, 4))
ar4 = np.ones((3, 5))
print(ar1)
print(ar2)
print(ar3)
print(ar4)
```

```python
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
[[10 11 12 13]
 [14 15 16 17]
 [18 19 20 21]]
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
[[1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]]
```

```python
# 矩阵加法
print(ar1 + ar2)
print(ar1 + ar2 + ar3)
```

```python
[[10 12 14 16]
 [18 20 22 24]
 [26 28 30 32]]
[[11. 13. 15. 17.]
 [19. 21. 23. 25.]
 [27. 29. 31. 33.]]
```

```python
ar1 * 10
```

```python
array([[  0,  10,  20,  30],
       [ 40,  50,  60,  70],
       [ 80,  90, 100, 110]])
```

```python
# 数组与矩阵相乘

print(ar1 * ar2)
print('--------------------')
# 数组相乘 → numpy里面两个shape相同的数组可以直接相乘，对应位置的值的乘积为结果
# 如果shape不同，则报错

a1 = np.array([2, 3, 4])
b1 = np.array([5, 6, 7]).reshape(3, 1)
# b1 = np.array([5, 6, 7])
c1 = np.dot(a1, b1)
print(a1.shape, b1.shape, c1.shape, c1)
print(c1, type(c1))

a2 = np.array([
    [1, 2, 3],
    [2, 3, 4]
])
b2 = np.array([
    [4, 4],
    [5, 5],
    [6, 6]
])
c2 = np.dot(a2, b2)
print(a2.shape, b2.shape, c2.shape)
print(c2)
```

```python
[[  0  11  24  39]
 [ 56  75  96 119]
 [144 171 200 231]]
--------------------
(3,) (3, 1) (1,) [56]
[56] <class 'numpy.ndarray'>
(2, 3) (3, 2) (2, 2)
[[32 32]
 [47 47]]
```

```python
# 矩阵乘法
a3 = np.array([
    [-2, 4],
    [1, -2]
])
b3 = np.array([
    [2, 4],
    [-3, -6]
])
print(np.dot(a3, b3))
print(np.dot(b3, a3))
```

```python
[[-16 -32]
 [  8  16]]
[[0 0]
 [0 0]]
```

```python
# 矩阵的转置
A = np.array([
    [2, 0, -1],
    [1, 3, 2]
])
B = np.array([
    [1, 7, -1],
    [4, 2, 3],
    [2, 0, 1]
])
np.dot(A, B).T
```

```python
array([[ 0, 17],
       [14, 13],
       [-3, 10]])
```

#### 2.2 逆矩阵

设A是数域上的一个n阶矩阵，若在相同数域上存在另一个n阶矩阵B，使得： AB=BA=E ，则我们称B是A的逆矩阵，而A则被称为可逆矩阵。注：E为单位矩阵 → 单位矩阵值为1

唯一性：若矩阵A是可逆的，则A的逆矩阵是唯一的

A的逆矩阵的逆矩阵还是A。记作（A-1）-1=A

可逆矩阵A的转置矩阵AT也可逆，并且（AT）-1=（A-1）T (转置的逆等于逆的转置）
两个可逆矩阵的乘积依然可逆

```python
A = np.array([
    [1, 2, 3],
    [2, 2, 1],
    [3, 4, 3]
])
print(A)
print(np.linalg.det(A))
```

```python
[[1 2 3]
 [2 2 1]
 [3 4 3]]
1.9999999999999991
```

```python
# numpy求逆矩阵 B → np.linalg.inv()

B = np.linalg.inv(A)
print(B)
print(np.linalg.det(B))
```

```python
[[ 1.   3.  -2. ]
 [-1.5 -3.   2.5]
 [ 1.   1.  -1. ]]
0.5000000000000004
```

```python
E = np.dot(A, B)
print(E)
print(np.linalg.det(E))
```

```python
[[ 1.00000000e+00  2.22044605e-16  0.00000000e+00]
 [-2.22044605e-16  1.00000000e+00  4.44089210e-16]
 [-2.22044605e-16  2.22044605e-16  1.00000000e+00]]
1.0
```

```python
A_bs = B*np.linalg.det(A)
print(A_bs)
print(np.linalg.det(A_bs))
```

```python
[[ 2.  6. -4.]
 [-3. -6.  5.]
 [ 2.  2. -2.]]
3.9999999999999947
```

```python
a = np.array([
    [4, 1, 2, 4],
    [1, 2, 0, 2],
    [1, 0, 2, 0],
    [0, 1, 1, 0]
])
print(np.linalg.inv(a))
np.linalg.det(a)
```

```python
[[-2.   4.   5.  -6. ]
 [-1.   2.   2.  -2. ]
 [ 1.  -2.  -2.   3. ]
 [ 2.  -3.5 -4.5  5. ]]





1.9999999999999993
```

```python
a = np.array([
        [1,1,1],
        [1,2,4],
        [1,3,9]
    ])
a1 = np.array([
        [2,1,1],
        [3,2,4],
        [5,3,9]
    ])
a2 = np.array([
        [1,2,1],
        [1,3,4],
        [1,5,9]
    ])
a3 = np.array([
        [1,1,2],
        [1,2,3],
        [1,3,5]
    ])
x1 = np.linalg.det(a1)/np.linalg.det(a)
x2 = np.linalg.det(a2)/np.linalg.det(a)
x3 = np.linalg.det(a3)/np.linalg.det(a)

print('该方程的三个根为：x1=%.2f, x2=%.2f, x3=%.2f' % (x1,x2,x3))
```

```python
该方程的三个根为：x1=2.00, x2=-0.50, x3=0.50
```

#### 2.3 线性方程组

```python
# 计算秩

a = np.array([
        [1,2,3],
        [2,3,-5],
        [4,7,1]
    ])
b = np.array([
        [3,2,0,5,0],
        [3,-2,3,6,-1],
        [2,0,1,5,-3],
        [1,6,-4,-1,4]
    ])
r1 = np.linalg.matrix_rank(a)
r2 = np.linalg.matrix_rank(b)

print(r1, r2)
```

```python
2 3
```

#### 2.4 向量的性质

* np.linalg.eigvals() → 计算矩阵的特征值
* np.linalg.eig() → 返回包含特征值和对应特征向量的元组

```python
a = np.array([
    [3, -1],
    [-1, 3]
])
print(np.linalg.eigvals(a))
print(np.linalg.eig(a))
```

```python
[4. 2.]
(array([4., 2.]), array([[ 0.70710678,  0.70710678],
       [-0.70710678,  0.70710678]]))
```

#### 2.5 矩阵的对角化

* np.diag() → 对角化

```python
a = np.array([
    [-2, 1, 1],
    [0, 2, 0],
    [-4, 1, 3]
])
print(np.linalg.eigvals(a))
np.diag(np.linalg.eigvals(a))
```

```python
[-1.  2.  2.]





array([[-1.,  0.,  0.],
       [ 0.,  2.,  0.],
       [ 0.,  0.,  2.]])
```




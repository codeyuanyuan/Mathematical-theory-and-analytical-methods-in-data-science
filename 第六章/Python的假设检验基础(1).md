### **单个总体均值检验**



#### **大样本 和 小样本**
* 根据不同的样本个数，选取不同的概率函数
```py
def cal_z_value(sample_count: int, alpha=0.05, sides=2):
    """
    计算z值的函数， 在计算均值检验和比例检验时，根据不同的样本量来计算z值
    @param sample_count: 样本数量
    @param alpha: 显著性水平
    @param sides: 检验类型, 默认值为 2 为双侧检验，可选值为 1 表示单侧检验
    @return: numpy.float64 z值
    """
    assert sides in (1, 2)
    cal_func, params = (st.t, [sample_count - 1, ]) if sample_count <= 30 else (st.norm, [0, 1])
    return cal_func.ppf(1 - alpha / sides, *tuple(params))
```

#### 总体均值检验

```py
def hypothesis_testing_mean(sample_count, x_bar,
                            mu, sigma_or_std, alpha=0.05, sides=2):
    """
    单个总体的均值检验
    @param sample_count: 样本的数量
    @param x_bar: 当前的样本均值
    @param mu: 期望的总体均值
    @param sigma_or_std: 期望的总体标准差 / 当前的样本标准差
    @param alpha: 显著性水平：假设正确时却被拒绝的可能性
    @param sides:是否是双侧检验，
    @return: bool 是否拒绝原假设
    """
    z_or_t = (x_bar - mu) / (sigma_or_std / sample_count ** 0.5)
    z_value = cal_z_value(sample_count, alpha=alpha, sides=sides)
    return abs(z_or_t) > abs(z_value)
```

#### **总体比例检验**
* 比例问题基本都使用大样本，而小样本个体对比例影响过大。
```py
def hypothesis_testing_p(sample_count, p_, pi0, alpha=0.05, sides=2):
    """
    单个总体的比例检验
    @param sample_count: int 样本的个数
    @param p_: 样本的指定条件所占比例
    @param pi0: 假设的总体的所占比例
    @param alpha: 显著性水平
    @param sides: 检验类型
    @return: 是否拒绝假设
    """
    assert sample_count > 30
    z_ = (p_ - pi0) / (pi0 * (1 - pi0) / sample_count) ** 0.5
    z_value = cal_z_value(sample_count=sample_count, alpha=alpha, sides=sides)
    return abs(z_) > abs(z_value)
```
#### **总体方差检验**
```py
def hypothesis_testing_var(sample_count, s_2, sigma_2, alpha=0.05, sides=2):
    """
    单个总体的方差检验
    @param sample_count: int 样本的个数
    @param s_2: 样本的方差
    @param sigma_2: 假设的总体的方差
    @param alpha: 显著性水平
    @param sides: 检验类型
    @return: 是否拒绝假设
    """
    ddof = sample_count - 1 # 样本的自由度
    chi2_ = ddof * s_2 / sigma_2
    chi2_min_value = st.chi2.ppf(alpha / sides, ddof)
    chi2_max_value = st.chi2.ppf(1 - alpha / sides, ddof)
    return chi2_max_value > chi2_ > chi2_min_value
```


### **两个总体的参数检验**



#### **两个总体的均值之差检验**
```py
def hypothesis_testing_mean_sub(
        sample_count_tuple: tuple,
        x_bar_tuple: tuple,
        mu_sub,
        sigma_or_std_tuple: tuple,
        alpha=0.05, sides=2):
    """
    两个总体均值之差的检验
    参数元组的顺序，必须与样本的顺序一致
    @param sample_count_tuple: 两个样本的数量的元组
    @param x_bar_tuple: 两个样本的均值元组
    @param mu_sub: 假设的总体均值之差
    @param sigma_or_std_tuple: 两个总体的标准差 / 两个样本的标准差 元组
    @param alpha: 显著性水平
    @param sides: 检验类型
    @return: 是否拒绝假设
    """
    sample_count = min(sample_count_tuple)
    z_or_t = ((x_bar_tuple[0] - x_bar_tuple[1]) - mu_sub) / \
             (sigma_or_std_tuple[0] ** 2 / sample_count_tuple[0] + sigma_or_std_tuple[1] ** 2 / sample_count_tuple[
                 1]) ** 0.5
    z_value = cal_z_value(sample_count, alpha=alpha, sides=sides)
    return abs(z_or_t) > abs(z_value)
```
#### **两个总体的比例之差检验**
```py
def hypothesis_testing_p_sub(
        sample_count_tuple: tuple,
        p_tuple: tuple,
        pi0_sub,
        alpha=0.05, sides=2
):
    """
    两个总体比例之差的假设检验
    @param sample_count_tuple: 两个样本的数量的元组
    @param p_tuple: 样本比例的元组
    @param pi0_sub: 假设的总体比例之差
    @param alpha: 显著性水平
    @param sides: 检验类型
    @return: 是否拒绝假设
    """
    assert min(sample_count_tuple) > 30
    z_ = (p_tuple[0] - p_tuple[1] - pi0_sub) / \
         ((
                 p_tuple[0] * (1 - p_tuple[0]) / sample_count_tuple[0] +
                 p_tuple[1] * (1 - p_tuple[1]) / sample_count_tuple[1]
         ) ** 0.5)
    z_value = cal_z_value(sample_count_tuple[0], alpha=alpha, sides=sides)
    return abs(z_) > abs(z_value)
```
#### **两个总体的方差比检验**
```py
def hypothesis_testing_var_ratio(
        sample_count_tuple: tuple,
        s_2_ratio,
        sigma_2_ratio,
        relation: str = '=',
        alpha=0.05, sides=2
):
    """
    两个总体的方差比的假设检验
    @param sample_count_tuple: 两个样本的数量的元组
    @param s_2_ratio: 两个样本的方差比
    @param sigma_2_ratio: 假设的两个总体的方差比
    @param relation: 关系，可选值 =, <, >. 默认为 =
    @param alpha: 显著性水平
    @param sides: 检验类型
    @return: 是否拒绝假设
    """
    assert relation in ('=', '<', '>')
    s_2_ratio = min((s_2_ratio, 1/s_2_ratio))
    sides = sides if relation == '=' else 1
    # 标准 F 分布区间（当假设比例为1，假设两个总体相等）
    z_max = st.f.ppf(1 - alpha / sides, sample_count_tuple[0] - 1, sample_count_tuple[1] - 1)  # 求 z0 值(F分布的z值
    z_min = st.f.ppf(alpha / sides, sample_count_tuple[0] - 1, sample_count_tuple[1] - 1)  # 求 z0 值(F分布的z值
    print(z_min, z_max)
    # 根据预测比例转换 z 值
    z_value_set = {i*sigma_2_ratio for i in [z_min, z_max]}
    if relation == '=':
        return ~(min(z_value_set) < s_2_ratio < max(z_value_set))
    elif relation == '>':
        return sigma_2_ratio < max(z_value_set)
    elif relation == '<':
        return sigma_2_ratio > min(z_value_set)
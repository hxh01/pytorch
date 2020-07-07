"""
create on july 03, 2020
numpy 的基础:
https://zhuanlan.zhihu.com/p/32242331

"""
# 1. Numpy 中 arange() 主要是用于生成数组, numpy.arange(start, stop, step, dtype = None)
# start 可选项, 默认从0开始. stop 必须项. step 可选项, 默认为1, 如果指定step,则还须要给出start
# dtype 输出数组的类型。 如果未给出dtype，则从其他输入参数推断数据类型。
# 值在半开区间 [开始，停止]内生成（换句话说，包括开始但不包括停止的区间）
import numpy as np
x = np.arange(0, 10, 1)
print(x)
y = x.reshape(2, 5)
print(y)

a = np.array([1, 2, 3])
print(a)

# 2. linspace/logspace
b = np.linspace(1, 5, 5)     # 生成首位是1，末位是5，含5个数的等差数列
print(b)

# 3. random rand
c1 = np.random.rand(3)       # 生成0到1之间的均匀一维数据3个
c2 = np.random.rand(2, 3)    # 生成0到1之间的均匀两行三列
print(c1, c2)
# random randint
d1 = np.random.randint(1, 10, size=(2, 3))  # 在1到10之间生存两行三列
print(d1)
# random randn
e1 = (np.random.randn(3, 2))  # 标准正态分布 三行两列

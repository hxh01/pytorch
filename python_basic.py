# 当GPU可用时,我们可以运行以下代码
# 我们将使用`torch.device`来将tensor移入和移出GPU
# import torch
#
# x = torch.randn(1)
# if torch.cuda.is_available():
#     device = torch.device("cuda")          # a CUDA device object
#     y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
#     x = x.to(device)                       # 或者使用`.to("cuda")`方法
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype
# print("aaa")
# 1.模块调用
from dog import Dog
d1 = Dog("tom", 10)
print(d1.name)

# 2.多变量赋值
A, B, C, D = 1, 2, 3, 4
#
print("HELLO", A, "good", B)
# 3.循环
for i in range(10):
    print(i)
# 4.条件判断

# 5. 数组, 列表, 元组, 集合与字典
# 数组, 同一种数据类型的集合
array = [1, 2, 3, 4, 5]
# 列表:（打了激素的数组）:可以存储任意数据类型的集合, 列表也可以嵌套
list1 = [1, 2, True, "hello"]
list2 = [1, 2, [1, 2]]
print(list1, list2)
# 列表的索引
print(list1[0])
# 列表切片, n:表示n之后的元素,即从n之后开始切
print(list1[3:])
# 列表切片, :-n, -表示从后面向前面切,
print(list1[:-1])
# 列表的翻转
print(list1[::-1])
# 元组的定义, 带了紧箍咒的列表, 定义了不可改变
# 元祖如果只有一个元素时, 记得加逗号, 否则输出的类型不是元祖, 是int型或是str型
t = (1, 2, 3, 'tar')
# 集合

# 字典是一个无序的数据集合，使用print输出字典的时候，通常输出的顺序和定义的顺序是不一致的
message = {
    "name": "tom",
    "age": "25"
}
print(message)

# 6. 类的定义与使用, 类与前后代码留有两行空格,如下:


class Circle(object):
    pi = 3.14  # 类属性

    def __init__(self, r):
        self.r = r

    def get_area(self):
        return self.r**2*self.pi


circle1 = Circle(1)
circle2 = Circle(2)
print('----未修改前-----')
print('pi=\t', Circle.pi)
print('circle1.pi=\t', circle1.pi)  # 3.14
print('circle2.pi=\t', circle2.pi)  # 3.14
print('----通过类名修改后-----')
Circle.pi = 3.14159  # 通过类名修改类属性，所有实例的类属性被改变
print('pi=\t', Circle.pi)  # 3.14159
print('circle1.pi=\t', circle1.pi)  # 3.14159
print('circle2.pi=\t', circle2.pi)  # 3.14159
print('----通过circle1实例名修改后-----')
circle1.pi = 3.14111  # 实际上这里是给circle1创建了一个与类属性同名的实例属性
print('pi=\t', Circle.pi)  # 3.14159
print('circle1.pi=\t', circle1.pi)  # 实例属性的访问优先级比类属性高，所以是3.14111
print('circle2.pi=\t', circle2.pi)  # 3.14159
print('----删除circle1实例属性pi-----')

# 7.

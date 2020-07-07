"""
https://www.runoob.com/numpy/numpy-matplotlib.html
"""
import numpy as np
import matplotlib.pyplot as plt

# 1. plot show
# plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
# plt.axis([0, 6, 0, 20])
# plt.show()

# 2. plot title label
# x = np.arange(-10, 10, 1)
# y = x*x + 2
# plt.title("matplotlib demo")
# plt.xlabel("x axis caption")
# plt.ylabel("y axis caption")
# plt.plot(x, y)
# plt.plot(x, y, "or")   # 第三个参数设置x,y属性, o表示圆点,r代表颜色
# plt.show()

# 3. subplot
# 计算正弦和余弦曲线上的点的 x 和 y 坐标
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
# 建立 subplot 网格，高为 2，宽为 1
# 激活第一个 subplot
plt.subplot(2,  1,  1)
# 绘制第一个图像
plt.plot(x, y_sin)
plt.title('Sine')
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
# 将第二个 subplot 激活，并绘制第二个图像
plt.subplot(2,  1,  2)
plt.plot(x, y_cos)
plt.title('Cosine')
plt.xlabel("x axis")
plt.ylabel("y axis")
# 展示图像
plt.show()

# bar 生成条形图
x = [5, 8, 10]
y = [12, 16, 6]
x2 = [6, 9, 11]
y2 = [6, 15, 7]
plt.bar(x, y, align='center')
plt.bar(x2, y2, color='r', align='center')
plt.title('Bar graph')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()

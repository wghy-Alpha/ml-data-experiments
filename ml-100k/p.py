import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 直接写入4.txt中的数据
intervals = ['5-10', '5-15', '5-20', '5-25', '5-30', '5-35', '5-40', '5-45', '5-50']
cos_means = [0.607, 0.577, 0.596, 0.609, 0.611, 0.646, 0.665, 0.687, 0.704]
kl_means = [0.827, 1.162, 1.281, 1.417, 1.566, 1.566, 1.567, 1.519, 1.541]

plt.figure(figsize=(8, 5))
plt.plot(intervals, cos_means, marker='o', label='余弦相似度 长尾-热门')
plt.plot(intervals, kl_means, marker='s', label='KL散度 长尾-热门')
plt.xlabel('区间')
plt.ylabel('均值')
plt.title('不同区间下 长尾-热门 的均值变化')
plt.legend()
plt.tight_layout()
plt.show()
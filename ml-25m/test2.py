import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 横坐标（百分比）
percents = [5, 10, 15, 20, 25, 30, 35, 40, 45]
# 余弦相似度均值
cos_means = [0.6477, 0.6587, 0.6699, 0.6779, 0.6902, 0.6880, 0.7135, 0.7262, 0.7337]
plt.figure(figsize=(8, 5))
plt.plot(percents, cos_means, marker='o', label='余弦相似度均值')
plt.xlabel('随机比例（%）')
plt.ylabel('值')
plt.title('随机 余弦相似度与KL散度均值')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
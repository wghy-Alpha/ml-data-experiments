import matplotlib.pyplot as plt

# 直接写入数据
x_labels = [
    '5-10', '5-15', '5-20', '5-25', '5-30',
    '5-35', '5-40', '5-45', '5-50'
]
cos_vals = [
    0.5180, 0.5267, 0.5437, 0.5535, 0.5712,
    0.5924, 0.6146, 0.6362, 0.6586
]
kl_vals = [
    1.4774, 1.9114, 2.1467, 2.4287, 2.5881,
    2.6486, 2.6111, 2.5403, 2.3986
]

plt.figure(figsize=(8,5))
plt.plot(x_labels, cos_vals, marker='o', label='余弦相似度均值')
plt.plot(x_labels, kl_vals, marker='s', label='KL散度均值')
plt.xlabel('区间')
plt.ylabel('值')
plt.title('热门-长尾 余弦相似度与KL散度')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
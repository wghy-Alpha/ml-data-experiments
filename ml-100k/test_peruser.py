import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from collections import Counter
from scipy.special import rel_entr
import random

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
plt.style.use('ggplot')

# 加载数据集（假设使用MovieLens 100k数据集）
ratings = pd.read_csv('ml-100k/u.data', sep='\t', 
                      names=['user_id', 'item_id', 'rating', 'timestamp'])
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                    names=['item_id', 'title', 'release_date', 'video_release_date',
                          'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                          'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                          'Thriller', 'War', 'Western'])

# 合并评分和电影信息
data = pd.merge(ratings, movies, on='item_id')

# 只统计评分大于等于4分的电影
high_rating_data = data[data['rating'] >= 4]

# 统计每个物品的交互数（只统计交互数>0的物品）
item_popularity = high_rating_data['item_id'].value_counts().reset_index()
item_popularity.columns = ['item_id', 'interaction_count']

# 只保留交互数大于0的物品
item_popularity = item_popularity[item_popularity['interaction_count'] > 0]



genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# 构建物品交互量字典，便于后续查找
item_interaction_dict = dict(zip(item_popularity['item_id'], item_popularity['interaction_count']))

user_tail_interest = {}
user_hot_interest = {}
user_all_interest = {}

def softmax(x):
    x = np.array(x)
    e_x = np.exp(x)
    return e_x / e_x.sum()

for user_id, group in high_rating_data.groupby('user_id'):
    user_items = group['item_id'].unique()
    # 获取这些物品的全局交互量
    user_item_counts = [(iid, item_interaction_dict.get(iid, 0)) for iid in user_items]
    # 按交互量排序
    user_item_counts.sort(key=lambda x: x[1])
    n = len(user_item_counts)
    if n < 20:
        continue  # 交互物品太少跳过

    # 新定义：前5%-10%为热门，后90%-95%为长尾
    hot_start = int(n * 0.05)
    hot_end = max(hot_start + 1, int(n * 0.25))
    tail_start = int(n * 0.75)
    tail_end = max(tail_start + 1, int(n * 0.95))

    # 热门物品：前5%-10%
    hot_items = set([iid for iid, _ in user_item_counts[hot_start:hot_end]])
    # 长尾物品：后90%-95%
    tail_items = set([iid for iid, _ in user_item_counts[tail_start:tail_end]])

    # 该用户交互过的长尾物品
    tail_group = group[group['item_id'].isin(tail_items)]
    if len(tail_group) <= 2:
        continue
    # 该用户交互过的热门物品
    hot_group = group[group['item_id'].isin(hot_items)]
    if len(hot_group) == 0:
        continue
    # 计算兴趣向量（均值并softmax归一化）
    tail_interest = tail_group[genre_cols].sum()
    hot_interest = hot_group[genre_cols].sum()
    all_interest = group[genre_cols].sum()
    # softmax归一化
    tail_interest = pd.Series(softmax(tail_interest), index=genre_cols)
    hot_interest = pd.Series(softmax(hot_interest), index=genre_cols)
    all_interest = pd.Series(softmax(all_interest), index=genre_cols)
    user_tail_interest[user_id] = tail_interest
    user_hot_interest[user_id] = hot_interest
    user_all_interest[user_id] = all_interest

tail_hot_sim = []
tail_all_sim = []
hot_all_sim = []



for user_id in user_tail_interest:
    tail_vec = user_tail_interest[user_id].values
    hot_vec = user_hot_interest[user_id].values
    all_vec = user_all_interest[user_id].values

    # 余弦相似度
    sim_tail_hot = cosine_similarity([tail_vec], [hot_vec])[0][0]
    sim_tail_all = cosine_similarity([tail_vec], [all_vec])[0][0]
    sim_hot_all = cosine_similarity([hot_vec], [all_vec])[0][0]
    tail_hot_sim.append(sim_tail_hot)
    tail_all_sim.append(sim_tail_all)
    hot_all_sim.append(sim_hot_all)



# print(f"\n[余弦相似度] 长尾-热门: 均值={np.mean(tail_hot_sim):.3f}")


# # 绘制余弦相似度的柱形图
# import matplotlib.pyplot as plt

# sim_means = [
#     np.mean(tail_hot_sim),
#     np.mean(tail_all_sim),
#     np.mean(hot_all_sim)
# ]
# labels = ['Tail-Hot', 'Tail-All', 'Hot-All']


# # 绘制每个用户的余弦相似度分布柱形图（直方图）
# plt.figure(figsize=(10, 5))
# plt.hist(tail_hot_sim, bins=20, alpha=0.7, label='Tail-Hot')
# plt.hist(tail_all_sim, bins=20, alpha=0.7, label='Tail-All')
# plt.hist(hot_all_sim, bins=20, alpha=0.7, label='Hot-All')
# plt.xlabel('Cosine Similarity')
# plt.ylabel('Number of Users')
# plt.title('Distribution of Cosine Similarity between Interest Distributions')
# plt.legend()
# plt.show()

# 计算KL散度的函数（加极小值防止除零）
def kl_divergence(p, q):
    p = np.asarray(p) + 1e-12
    q = np.asarray(q) + 1e-12
    return np.sum(rel_entr(p, q))

tail_hot_kl = []
tail_all_kl = []
hot_all_kl = []

for user_id in user_tail_interest:
    tail_vec = user_tail_interest[user_id].values
    hot_vec = user_hot_interest[user_id].values
    all_vec = user_all_interest[user_id].values

    # KL散度（P||Q）
    tail_hot_kl.append(kl_divergence(hot_vec, tail_vec))
    tail_all_kl.append(kl_divergence(all_vec, tail_vec))
    hot_all_kl.append(kl_divergence(all_vec, hot_vec))

print(f"\n[余弦相似度] 长尾-热门: 均值={np.mean(tail_hot_sim):.3f} [KL散度] 长尾-热门: 均值={np.mean(tail_hot_kl):.3f}")

# # 可视化KL散度分布
# plt.figure(figsize=(10, 5))
# plt.hist(tail_hot_kl, bins=20, alpha=0.7, label='Tail-Hot')
# plt.hist(tail_all_kl, bins=20, alpha=0.7, label='Tail-All')
# plt.hist(hot_all_kl, bins=20, alpha=0.7, label='Hot-All')
# plt.xlabel('KL Divergence')
# plt.ylabel('Number of Users')
# plt.title('Distribution of KL Divergence between Interest Distributions')
# plt.legend()
# plt.show()





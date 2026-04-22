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

# 对所有物品按交互数排序，定义热门和长尾物品
item_popularity_sorted = item_popularity.sort_values('interaction_count', ascending=False)
hot_num = max(1, int(len(item_popularity_sorted) * 0.02))  # 前2%
tail_num = max(1, int(len(item_popularity_sorted) * 0.4))   # 后40%
hot_items = item_popularity_sorted.head(hot_num)['item_id'].tolist()
long_tail_items = item_popularity_sorted.tail(tail_num)['item_id'].tolist()

print(f"热门物品数量：{len(hot_items)}")
print(f"长尾物品数量：{len(long_tail_items)}")

user_tail_interest = {}
user_hot_interest = {}
user_all_interest = {}

def softmax(x):
    x = np.array(x)
    e_x = np.exp(x)
    return e_x / e_x.sum()

for user_id, group in high_rating_data.groupby('user_id'):
    tail_group = group[group['item_id'].isin(long_tail_items)]
    hot_group = group[group['item_id'].isin(hot_items)]
    if len(tail_group) == 0 or len(hot_group) == 0:
        continue
    all_interest = group[genre_cols].sum()
    tail_interest = tail_group[genre_cols].sum()
    hot_interest = hot_group[genre_cols].sum()
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



print(f"\n[余弦相似度] 长尾-热门: 均值={np.mean(tail_hot_sim):.3f}  长尾-总体: 均值={np.mean(tail_all_sim):.3f}  热门-总体: 均值={np.mean(hot_all_sim):.3f}")


# 绘制余弦相似度的柱形图
import matplotlib.pyplot as plt

sim_means = [
    np.mean(tail_hot_sim),
    np.mean(tail_all_sim),
    np.mean(hot_all_sim)
]
labels = ['Tail-Hot', 'Tail-All', 'Hot-All']


# 绘制每个用户的余弦相似度分布柱形图（直方图）
plt.figure(figsize=(10, 5))
plt.hist(tail_hot_sim, bins=20, alpha=0.7, label='Tail-Hot')
plt.hist(tail_all_sim, bins=20, alpha=0.7, label='Tail-All')
plt.hist(hot_all_sim, bins=20, alpha=0.7, label='Hot-All')
plt.xlabel('Cosine Similarity')
plt.ylabel('Number of Users')
plt.title('Distribution of Cosine Similarity between Interest Distributions')
plt.legend()
plt.show()

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

print(f"\n[KL散度] 长尾-热门: 均值={np.mean(tail_hot_kl):.3f}  长尾-总体: 均值={np.mean(tail_all_kl):.3f}  热门-总体: 均值={np.mean(hot_all_kl):.3f}")

# 可视化KL散度分布
plt.figure(figsize=(10, 5))
plt.hist(tail_hot_kl, bins=20, alpha=0.7, label='Tail-Hot')
plt.hist(tail_all_kl, bins=20, alpha=0.7, label='Tail-All')
plt.hist(hot_all_kl, bins=20, alpha=0.7, label='Hot-All')
plt.xlabel('KL Divergence')
plt.ylabel('Number of Users')
plt.title('Distribution of KL Divergence between Interest Distributions')
plt.legend()
plt.show()



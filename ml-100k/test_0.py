import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from collections import Counter

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

# 定义长尾物品和热门物品（后10%为长尾，前10%为热门）
tail_threshold = item_popularity['interaction_count'].quantile(0.1)
hot_threshold = item_popularity['interaction_count'].quantile(0.9)

long_tail_items = item_popularity[item_popularity['interaction_count'] <= tail_threshold]['item_id'].tolist()
hot_items = item_popularity[item_popularity['interaction_count'] >= hot_threshold]['item_id'].tolist()

genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

user_tail_interest = {}
user_hot_interest = {}
user_all_interest = {}

for user_id, group in high_rating_data.groupby('user_id'):
    # 该用户交互过的长尾物品
    tail_group = group[group['item_id'].isin(long_tail_items)]
    if len(tail_group) == 0:
        continue  # 跳过没有长尾物品交互的用户
    hot_group = group[group['item_id'].isin(hot_items)]
    # 计算兴趣向量（均值并归一化）
    tail_interest = tail_group[genre_cols].mean()
    hot_interest = hot_group[genre_cols].mean() if len(hot_group) > 0 else np.zeros(len(genre_cols))
    all_interest = group[genre_cols].mean()
    # 归一化
    if tail_interest.sum() > 0:
        tail_interest = tail_interest / tail_interest.sum()
    if hot_interest.sum() > 0:
        hot_interest = hot_interest / hot_interest.sum()
    if all_interest.sum() > 0:
        all_interest = all_interest / all_interest.sum()
    user_tail_interest[user_id] = tail_interest
    user_hot_interest[user_id] = hot_interest
    user_all_interest[user_id] = all_interest

tail_hot_sim = []
tail_all_sim = []
hot_all_sim = []

tail_hot_auc = []
tail_all_auc = []
hot_all_auc = []

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

    # AUC（用all_vec作为“真实分布”，tail_vec和hot_vec作为“预测分布”）
    # 由于AUC要求标签有0和1，这里用all_vec的均值做二值化
    y_true = (all_vec > all_vec.mean()).astype(int)
    if len(np.unique(y_true)) < 2:
        continue  # 跳过全0或全1的情况
    try:
        auc_tail = roc_auc_score(y_true, tail_vec)
        auc_hot = roc_auc_score(y_true, hot_vec)
        auc_tail_hot = roc_auc_score((tail_vec > tail_vec.mean()).astype(int), hot_vec)
        tail_all_auc.append(auc_tail)
        hot_all_auc.append(auc_hot)
        tail_hot_auc.append(auc_tail_hot)
    except Exception:
        continue

print(f"\n[余弦相似度] 长尾-热门: 均值={np.mean(tail_hot_sim):.3f}  长尾-总体: 均值={np.mean(tail_all_sim):.3f}  热门-总体: 均值={np.mean(hot_all_sim):.3f}")
print(f"[AUC] 长尾-总体: 均值={np.mean(tail_all_auc):.3f}  热门-总体: 均值={np.mean(hot_all_auc):.3f}  长尾-热门: 均值={np.mean(tail_hot_auc):.3f}")





